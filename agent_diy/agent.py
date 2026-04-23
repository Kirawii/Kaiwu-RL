#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

峡谷追猎 - DIY Agent 主类
完全参考 agent_target_dqn 亚军方案
使用DQN算法 + SimbaV2网络
"""

import torch
import numpy as np

# 设置PyTorch线程数
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm_dqn import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData, NumpyData2SampleData
from agent_diy.feature.preprocessor import Preprocessor


class Agent(BaseAgent):
    """
    DIY Agent 主类 - DQN版本
    完全复制 agent_target_dqn 的设计
    """

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.device = device
        self.logger = logger
        self.monitor = monitor
        self.agent_type = agent_type

        # 初始化预处理器
        self.preprocessor = Preprocessor()

        # 初始化算法 (包含模型)
        self.algorithm = Algorithm(device, logger, monitor)

        # 上一步动作
        self.last_action = -1

        # 第一加载标记 (用于target网络初始化)
        self.first_load = True

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """每局开始时重置状态"""
        self.preprocessor.reset()
        self.last_action = -1

    def observation_process(self, env_obs, preprocessor=None, extra_info=None):
        """
        将原始环境观测转换为模型输入特征

        Args:
            env_obs: 环境原始观测
            preprocessor: 预处理器 (未使用，使用self.preprocessor)
            extra_info: 额外信息

        Returns:
            obs_data: ObsData (feature, legal_act)
            remain_info: 用于奖励计算的信息
        """
        result = self.preprocessor.feature_process(env_obs, self.last_action)
        feature, legal_action, remain_info, map_tensor = result

        obs_data = ObsData(
            feature=feature,
            legal_act=legal_action,
        )

        return obs_data, remain_info

    def predict(self, list_obs_data):
        """
        训练时的动作预测 (带epsilon-greedy探索)

        Args:
            list_obs_data: ObsData 列表

        Returns:
            ActData 列表
        """
        # 调用algorithm的predict_detail方法
        act_data_list = self.algorithm.predict_detail(list_obs_data, exploit_flag=False)

        # 转换为ActData格式 (包含move_dir和use_talent)
        results = []
        for act_data in act_data_list:
            # move_dir: 0-7, use_talent: 0或1
            action = act_data.move_dir + act_data.use_talent * 8
            results.append(ActData(
                action=[action],
                d_action=[action],  # DQN没有d_action概念，用相同值
                prob=[0.0] * Config.ACTION_NUM,  # DQN不使用概率
                value=[0.0],  # DQN在predict时不计算value
            ))

        return results

    def exploit(self, env_obs):
        """
        评估时的动作预测 (纯利用，无探索)

        Args:
            env_obs: 环境观测

        Returns:
            int: 动作索引
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data_list = self.algorithm.predict_detail([obs_data], exploit_flag=True)

        # 转换动作格式
        act_data = act_data_list[0]
        action = act_data.move_dir + act_data.use_talent * 8
        self.last_action = action

        return action

    def learn(self, list_sample_data):
        """
        训练模型

        Args:
            list_sample_data: numpy数组列表 (由SampleData转换而来)
        """
        if list_sample_data is None or len(list_sample_data) == 0:
            return

        # 将numpy数组转换回SampleData对象
        if isinstance(list_sample_data[0], np.ndarray):
            sample_data_list = [NumpyData2SampleData(frame) for frame in list_sample_data]
        else:
            sample_data_list = list_sample_data

        self.algorithm.learn(sample_data_list)

    def save_model(self, path=None, id="1"):
        """
        保存模型检查点

        Args:
            path: 保存路径
            id: 模型ID
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {
            k: v.clone().cpu() for k, v in self.algorithm.model.state_dict().items()
        }
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """
        加载模型检查点

        Args:
            path: 模型路径
            id: 模型ID
        """
        import os
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        if not os.path.exists(model_file_path):
            self.logger.warning(f"model file not found: {model_file_path}")
            return

        try:
            self.algorithm.model.load_state_dict(
                torch.load(model_file_path, map_location=self.device)
            )

            # 第一次加载时更新target网络
            if self.first_load:
                self.algorithm.update_target_q()
                self.first_load = False
                self.logger.info(f"First load model {model_file_path} and update target q successfully")
            else:
                self.logger.info(f"load model {model_file_path} successfully")
        except Exception as e:
            self.logger.error(f"load model failed: {e}")

    def action_process(self, act_data, is_stochastic=True):
        """
        将 ActData 转换为环境可执行的动作

        Args:
            act_data: ActData
            is_stochastic: 是否随机采样 (DQN中由epsilon-greedy处理)

        Returns:
            int: 动作索引
        """
        # act_data.action已经是0-15的索引
        action = int(act_data.action[0]) if isinstance(act_data.action, list) else int(act_data.action)
        self.last_action = action
        return action
