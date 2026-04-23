#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

峡谷追猎 - DIY Agent 训练工作流
优化版本：课程学习 + 情景奖励 + 动态参数
"""

import os
import time
import numpy as np

from agent_diy.feature.definition import (
    SampleData,
    reward_shaping,
    SampleData2NumpyData,
)
from agent_diy.conf.conf import Config
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery

# 可视化工具（可选）
try:
    from agent_diy.utils.debug_utils import EpisodeLogger, SimpleMapVisualizer
    DEBUG_UTILS_AVAILABLE = True
except ImportError:
    DEBUG_UTILS_AVAILABLE = False


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """
    训练工作流主函数

    Args:
        envs: 环境列表
        agents: Agent列表
        logger: 日志器
        monitor: 监控器
    """
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # 读取用户配置
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    # 创建对局运行器
    # 调试模式开关：设置为True启用可视化工具
    enable_debug = False  # 关闭调试/可视化
    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
        enable_debug=enable_debug,
    )

    # 训练循环
    while True:
        for g_data in episode_runner.run_episodes():
            # 发送样本进行训练 (保持SampleData格式)
            agent.send_sample_data(g_data)
            g_data.clear()

            # 定期保存模型
            now = time.time()
            if now - last_save_model_time >= 1800:  # 每30分钟
                agent.save_model()
                last_save_model_time = now

                # 打印训练步数
                train_step = agent.algorithm.train_step if hasattr(agent.algorithm, 'train_step') else 0
                logger.info(f"[Train] step:{train_step} model saved")


class EpisodeRunner:
    """
    对局运行器 - 支持课程学习和情景奖励
    """

    def __init__(self, env, agent, usr_conf, logger, monitor, enable_debug=False):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.enable_debug = enable_debug and DEBUG_UTILS_AVAILABLE

        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

        # 初始化调试日志记录器
        if self.enable_debug:
            self.episode_logger = EpisodeLogger(log_dir="./episode_logs")
        else:
            self.episode_logger = None

    def run_episodes(self):
        """
        运行单局游戏并yield收集的样本
        """
        while True:
            # 定期获取训练指标
            self._check_training_metrics()

            # 重置环境
            env_obs = self.env.reset(self.usr_conf)

            # 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # 重置Agent并加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # 初始观测处理
            obs_data, remain_info = self.agent.observation_process(env_obs)

            # 样本收集器
            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0

            # 情景统计
            escape_count = 0
            flash_waste_count = 0
            treasure_collected = 0

            self.logger.info(f"Episode {self.episode_cnt} start")

            # 重置调试日志记录器
            if self.episode_logger:
                self.episode_logger.reset()

            while not done:
                # Agent推理
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                # 与环境交互
                env_reward, env_obs = self.env.step(act)

                # 容灾处理
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                # 检查游戏状态
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # 处理下一步观测
                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                # 计算奖励塑形
                reward = reward_shaping(
                    frame_no=step,
                    score=env_reward.get("reward", 0),
                    terminated=terminated,
                    truncated=truncated,
                    remain_info=remain_info,
                    _remain_info=_remain_info,
                    obs=obs_data,
                    _obs=_obs_data,
                )

                if reward is None:
                    reward = 0.0

                total_reward += float(reward)

                # 记录调试信息
                if self.episode_logger:
                    debug_info = self.episode_logger.log_step(env_obs, reward, act, remain_info)
                    # 每50步打印一次位置信息
                    if step % 50 == 0:
                        self.logger.info(
                            f"[DEBUG] Step {debug_info['step']}: "
                            f"Hero={debug_info['hero']}, "
                            f"T={debug_info['treasures']}, B={debug_info['buffs']}, "
                            f"Danger={debug_info['danger']}, Reward={debug_info['reward']}"
                        )

                # 统计情景
                if _remain_info.get('escaped', False):
                    escape_count += 1
                if _remain_info.get('flash_wasted', False):
                    flash_waste_count += 1
                if _remain_info.get('treasure_collected', 0) > treasure_collected:
                    treasure_collected = _remain_info.get('treasure_collected', 0)

                # 终局奖励 (标量)
                final_reward = 0.0
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)
                    treasures = env_info.get("treasures_collected", 0)
                    flash_count = env_info.get("flash_count", 0)

                    if terminated:
                        final_reward = Config.DEATH_PENALTY
                        result_str = "FAIL"
                    else:
                        final_reward = Config.WIN_REWARD
                        result_str = "WIN"

                    # 获取课程学习阶段
                    progress_ratio = remain_info.get('progress_ratio', 0.5)
                    phase = "EARLY" if remain_info.get('is_early_phase') else (
                        "LATE" if remain_info.get('is_late_phase') else "MID"
                    )

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} score:{total_score:.1f} "
                        f"treasures:{treasures} flash:{flash_count} "
                        f"escapes:{escape_count} phase:{phase} "
                        f"reward:{total_reward:.3f}"
                    )

                # 构造样本帧 (DQN格式, 与agent_target_dqn一致)
                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    _obs=None,  # 将在下一帧填充
                    obs_legal=np.array(obs_data.legal_act, dtype=np.float32),
                    _obs_legal=None,  # 将在下一帧填充
                    act=int(act_data.action[0]),
                    rew=float(reward[0]) if isinstance(reward, np.ndarray) else float(reward),
                    ret=0.0,  # 备用字段
                    done=float(done),
                )
                collector.append(frame)

                # 对局结束处理
                if done:
                    if collector:
                        # 添加终局奖励到最后一帧
                        current_reward = collector[-1].rew
                        if isinstance(current_reward, np.ndarray):
                            collector[-1].rew = current_reward + final_reward
                        else:
                            collector[-1].rew = float(current_reward) + final_reward

                        # 填充最后一帧的next_obs (用自身表示终局)
                        collector[-1]._obs = collector[-1].obs
                        collector[-1]._obs_legal = collector[-1].obs_legal

                    # 上报监控
                    self._report_monitor(total_reward + final_reward, step, treasure_collected, escape_count)

                    # 直接yield收集的样本 (DQN不需要GAE处理)
                    if collector:
                        yield collector

                    # 保存调试日志（前10局和每100局）
                    if self.episode_logger and (self.episode_cnt <= 10 or self.episode_cnt % 100 == 0):
                        log_path = self.episode_logger.save_episode()
                        self.logger.info(f"[DEBUG] Episode log saved: {log_path}")

                    break

                # 填充上一帧的next_obs和next_legal_action
                if collector:
                    collector[-1]._obs = np.array(_obs_data.feature, dtype=np.float32)
                    collector[-1]._obs_legal = np.array(_obs_data.legal_act, dtype=np.float32)

                # 更新状态
                obs_data = _obs_data
                remain_info = _remain_info

    def _check_training_metrics(self):
        """定期检查训练指标"""
        now = time.time()
        if now - self.last_get_training_metrics_time >= 60:
            training_metrics = get_training_metrics()
            self.last_get_training_metrics_time = now
            if training_metrics is not None:
                self.logger.info(f"training_metrics is {training_metrics}")

    def _report_monitor(self, total_reward, step, treasures, escapes):
        """上报监控数据"""
        now = time.time()
        if now - self.last_report_monitor_time >= 60 and self.monitor:
            monitor_data = {
                "reward": round(float(total_reward), 4),
                "episode_steps": step,
                "episode_cnt": self.episode_cnt,
                "treasures_collected": treasures,
                "escape_count": escapes,
            }
            self.monitor.put_data({os.getpid(): monitor_data})
            self.last_report_monitor_time = now
