#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np
import time
import os
from kaiwu_agent.utils.common_func import Frame, attached
from agent_ppo.utils.record_metrics import RecordMetrics

from tools.train_env_conf_validate import read_usr_conf
from agent_ppo.feature.definition import (
    SampleManager,
)
from agent_ppo.agent import Agent
from agent_ppo.conf.conf import Config as cfg
from tools.metrics_utils import get_training_metrics

metrics = {
    'win_rate': RecordMetrics(max_len=50),
    'reward': RecordMetrics(),  # 每个episode都reset
}

@attached
def workflow(envs, agents, logger=None, monitor=None):
    try:
        env, agent = envs[0], agents[0]
        episode_num_every_epoch = 1
        last_save_model_time = 0
        last_put_data_time = 0
        monitor_data = {}

        # Read and validate configuration file
        # 配置文件读取和校验
        usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error(f"usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
            return

        while True:
            for g_data, monitor_data in run_episodes(episode_num_every_epoch, env, agent, usr_conf, logger, monitor):
                agent.learn(g_data)
                g_data.clear()
                # exit()

            # Save model file
            # 保存model文件
            now = time.time()
            if now - last_save_model_time >= 1800:  # 30 mins
                agent.save_model()
                last_save_model_time = now

            # Report monitoring metrics
            # 上报监控指标
            if now - last_put_data_time >= 60 and monitor:
                monitor.put_data({os.getpid(): monitor_data})
                last_put_data_time = now

    except Exception as e:
        raise RuntimeError(f"workflow error, {e}") from e


def run_episodes(n_episode, env, agent: Agent, usr_conf, logger, monitor):
    try:
        for episode in range(n_episode):
            collector = SampleManager()
            total_reward = 0
            total_hit_wall = 0
            metrics['reward'].reset()

            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                logger.info(f"training_metrics is {training_metrics}")

            # Reset the task and get the initial state
            # 重置任务, 并获取初始状态
            treasure_num = int(np.random.choice(14, p=cfg.TREASURE_NUM_PROB))
            # # v0.7.1 固定8宝箱位置, 测试固定宝箱状态是否有必要, 寻找宝箱奖励是否可行
            usr_conf['env_conf']['map_fish']['treasure_random'] = False
            usr_conf['env_conf']['map_fish']['start_random'] = False
            usr_conf['env_conf']['map_fish']['end_random'] = False
            usr_conf['env_conf']['map_fish']['obstacle_random '] = False

            usr_conf['env_conf']['map_fish']['treasure_count'] = treasure_num
            # print(f"{'='*10}[DEBUG] usr_conf: {usr_conf}{'='*10}")
            obs, extra_info = env.reset(usr_conf=usr_conf)
            if extra_info["result_code"] < 0:
                logger.error(
                    f"env.reset result_code is {extra_info['result_code']}, result_message is {extra_info['result_message']}"
                )
                raise RuntimeError(extra_info["result_message"])
            elif extra_info["result_code"] > 0:
                continue

            # At the start of each game, support loading the latest model file
            # The call will load the latest model from a remote training node
            # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
            agent.reset()
            agent.load_model(id="latest")

            terminated, truncated, done = False, False, False
            step = 0

            max_step_no = int(os.environ.get("max_step_no", "0"))

            while not done:
                # Feature processing
                # 特征处理
                obs_data = agent.observation_process(obs, terminated, truncated, extra_info)
                total_hit_wall += int(agent.state_manager.hit_wall)

                # Agent performs inference, gets the predicted action for the next frame
                # Agent 进行推理, 获取下一帧的预测动作
                act_data, model_version = agent.predict(list_obs_data=[obs_data])

                # Unpack ActData into action
                # ActData 解包成动作
                act = agent.action_process(act_data[0])

                # Interact with the environment, execute actions, get the next state
                # 与环境交互, 执行动作, 获取下一步的状态
                step_no, _obs, terminated, truncated, _extra_info = env.step(act)
                if _extra_info["result_code"] != 0:
                    logger.warning(
                        f"_extra_info.result_code is {_extra_info['result_code']}, \
                        _extra_info.result_message is {_extra_info['result_message']}"
                    )
                    break

                step += 1

                reward = obs_data.reward
                metrics['reward'].record(reward)
                total_reward += reward

                # Determine task over, and update the number of victories
                # 判断任务结束, 并更新胜利次数
                game_info = _extra_info["game_info"]
                if truncated:
                    metrics['win_rate'].record(False)
                    win_rate = metrics['win_rate'].get_average()
                    logger.info(
                        f"Game truncated! step_no:{step_no} score:{game_info['total_score']} win_rate:{win_rate}"
                    )
                elif terminated:
                    metrics['win_rate'].record(True)
                    win_rate = metrics['win_rate'].get_average()
                    logger.info(
                        f"Game terminated! step_no:{step_no} score:{game_info['total_score']} win_rate:{win_rate}"
                    )
                done = terminated or truncated or (max_step_no > 0 and step >= max_step_no)

                # If the task is over, the sample is processed and sent to training
                # 如果任务结束，则进行样本处理，将样本送去训练
                if done:
                    obs_data = agent.observation_process(obs, terminated, truncated, extra_info)
                    # print(f"{'='*10}[DEBUG] total_reward: {total_reward}{'='*10}")
                    monitor_data = {
                        "diy_1": win_rate,
                        "diy_2": metrics['reward'].get_average(),
                        "diy_3": total_reward,
                        "diy_4": total_hit_wall,
                    }
                    collector.process_last_frame(np.array([reward]))
                    if len(collector.samples) > 0:
                        yield collector.get_game_data(), monitor_data

                    break
                
                # Construct task frames to prepare for sample construction
                # 构造任务帧，为构造样本做准备
                collector.sample_process(
                    feature=obs_data.feature,
                    legal_action=obs_data.legal_action,
                    prob=[act_data[0].prob],
                    action=[act_data[0].action],
                    value=act_data[0].value,
                    reward=np.array([reward]),  # 保持shape都是 (1, ...)
                )


                # Status update
                # 状态更新
                obs = _obs
                extra_info = _extra_info

    except Exception as e:
        raise RuntimeError(f"run_episodes error, {e}") from e
