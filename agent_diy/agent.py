#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

峡谷追猎 - DIY Agent 主类
优化版本：GRU时序状态 + 完整动作空间 + 动态学习率
"""

import torch
import numpy as np

# 设置PyTorch线程数
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData, reward_shaping
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.model.model import Model


class Agent(BaseAgent):
    """
    DIY Agent 主类 - 优化版本
    实现与环境交互、特征处理、动作预测、模型训练
    支持GRU时序状态管理和动态学习率
    """

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        # 设置随机种子
        torch.manual_seed(0)
        np.random.seed(0)

        self.device = device
        self.logger = logger
        self.monitor = monitor
        self.agent_type = agent_type

        # 初始化模型
        self.model = Model(device).to(self.device)

        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # 学习率调度器 (可选)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=Config.LR_DECAY_RATE
        ) if Config.LR_DECAY else None

        # 初始化算法
        self.algorithm = Algorithm(
            self.model, self.optimizer, self.device, logger, monitor
        )

        # 初始化预处理器
        self.preprocessor = Preprocessor()

        # 上一步动作
        self.last_action = -1

        # GRU隐藏状态
        self.hidden_state = None

        # 训练样本收集
        self.sample_collector = []

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """
        每局开始时重置状态
        """
        self.preprocessor.reset()
        self.last_action = -1
        self.sample_collector = []

        # 重置GRU隐藏状态
        self.hidden_state = None
        self.model.reset_hidden_state()

    def observation_process(self, env_obs, preprocessor=None, extra_info=None):
        """
        将原始环境观测转换为模型输入特征

        Args:
            env_obs: 环境原始观测
            preprocessor: 预处理器 (这里使用 self.preprocessor)
            extra_info: 额外信息

        Returns:
            obs_data: ObsData (feature, legal_act)
            remain_info: 用于奖励计算的信息
        """
        result = self.preprocessor.feature_process(env_obs, self.last_action)

        # 解包返回结果 (现在包含map_tensor)
        if len(result) == 4:
            feature, legal_action, remain_info, map_tensor = result
        else:
            feature, legal_action, remain_info = result
            map_tensor = None

        obs_data = ObsData(
            feature=list(feature),
            legal_act=legal_action,
            map_tensor=map_tensor,
        )

        return obs_data, remain_info

    def predict(self, list_obs_data):
        """
        训练时的动作预测 (随机采样，用于探索)

        Args:
            list_obs_data: ObsData 列表

        Returns:
            ActData 列表
        """
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_action = obs_data.legal_act
        map_tensor = getattr(obs_data, 'map_tensor', None)

        # 运行模型
        logits, value, self.hidden_state, danger_pred, prob = self._run_model(
            feature, legal_action, map_tensor, self.hidden_state
        )

        # 随机采样动作 (训练时)
        action = self._legal_sample(prob, use_max=False)

        # 贪心动作 (用于评估)
        d_action = self._legal_sample(prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """
        评估时的动作预测 (贪心选择)

        Args:
            env_obs: 环境观测

        Returns:
            int: 动作
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """
        训练模型

        Args:
            list_sample_data: SampleData 列表
        """
        if not list_sample_data:
            return

        result = self.algorithm.learn(list_sample_data)

        # 更新学习率
        if self.scheduler is not None:
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr > Config.LR_MIN:
                self.scheduler.step()

        return result

    def save_model(self, path=None, id="1"):
        """
        保存模型检查点

        Args:
            path: 保存路径
            id: 模型ID
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {
            k: v.clone().cpu() for k, v in self.model.state_dict().items()
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
            self.model.load_state_dict(
                torch.load(model_file_path, map_location=self.device)
            )
            self.logger.info(f"load model {model_file_path} successfully")
        except Exception as e:
            self.logger.error(f"load model failed: {e}")

    def action_process(self, act_data, is_stochastic=True):
        """
        将 ActData 转换为环境可执行的动作

        Args:
            act_data: ActData
            is_stochastic: 是否随机采样

        Returns:
            int: 动作索引
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action, map_tensor=None, hidden_state=None):
        """
        运行模型推理

        Args:
            feature: 特征向量
            legal_action: 合法动作掩码
            map_tensor: 4通道51x51地图张量
            hidden_state: GRU隐藏状态

        Returns:
            logits: 动作logits
            value: 状态价值
            new_hidden_state: 更新的GRU隐藏状态
            danger_pred: 危险预测
            prob: 动作概率分布
        """
        self.model.set_eval_mode()

        # 转换为tensor
        obs_tensor = torch.tensor(
            np.array([feature]), dtype=torch.float32
        ).to(self.device)
        legal_action_np = np.array(legal_action, dtype=np.float32)

        # 转换map_tensor
        if map_tensor is not None:
            map_tensor_t = torch.tensor(
                np.array([map_tensor]), dtype=torch.float32
            ).to(self.device)
        else:
            map_tensor_t = None

        # 如果没有提供隐藏状态，获取初始状态
        if hidden_state is None:
            hidden_state = self.model.get_initial_hidden(batch_size=1, device=self.device)

        with torch.no_grad():
            logits, value, new_hidden_state, danger_pred, _, _ = self.model(
                obs_tensor, map_tensor=map_tensor_t, inference=True, hidden_state=hidden_state
            )

        # 转换为numpy
        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]
        danger_pred_np = danger_pred.cpu().numpy()[0]

        # 合法动作掩码softmax
        prob = self._legal_soft_max(logits_np, legal_action_np)

        return logits_np, value_np, new_hidden_state, danger_pred_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        """
        合法动作掩码下的softmax (numpy版)
        """
        _w, _e = 1e20, 1e-5
        tmp = input_hidden - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """
        从概率分布中采样动作

        Args:
            probs: 动作概率分布
            use_max: 是否选择概率最大的动作

        Returns:
            int: 动作索引
        """
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
