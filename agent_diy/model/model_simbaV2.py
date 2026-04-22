#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

峡谷追猎 - DIY Agent SimbaV2 神经网络模型
完全参考 agent_target_dqn 亚军方案
架构: Embedder -> Encoder (2 LERP blocks) -> Predictor
"""

import torch
import torch.nn as nn
import numpy as np

from agent_diy.conf.conf import Config
from agent_diy.model.simbaV2.agents.networks import SimbaV2Critic, l2normalize_network


class Model(nn.Module):
    """
    SimbaV2 DQN 网络模型
    使用分桶离散价值估计 (HyperDiscreteValueHead)

    输入特征维度: Config.FEATURE_DIM
    输出: Q值分布 (num_bins个离散值)
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_diy_simbaV2"
        self.device = device

        # 保存配置用于后续计算
        self.v_min = -100.0  # 最小Q值
        self.v_max = 100.0   # 最大Q值
        self.num_bins = Config.SIMBA_NUM_BINS  # 默认256

        # 计算原子值 (用于C51风格的分桶)
        self.register_buffer(
            "atoms",
            torch.linspace(self.v_min, self.v_max, self.num_bins)
        )

        # SimbaV2 Critic网络
        self.critic = SimbaV2Critic(
            in_dim=Config.FEATURE_DIM,
            num_blocks=Config.SIMBA_NUM_BLOCKS,      # 2个LERP块
            hidden_dim=Config.SIMBA_HIDDEN_DIM,      # 512
            scaler_init=Config.SIMBA_SCALER_INIT,    # sqrt(2/512)
            scaler_scale=Config.SIMBA_SCALER_SCALE,  # 1.0
            alpha_init=Config.SIMBA_ALPHA_INIT,      # 1/3
            alpha_scale=Config.SIMBA_ALPHA_SCALE,    # 1.0
            c_shift=Config.SIMBA_C_SHIFT,            # 3.0
            num_bins=Config.SIMBA_NUM_BINS,          # 256
        )

        # 温度参数 (用于探索)
        self.log_temp = nn.Parameter(torch.ones([]) * np.log(0.01))

    def forward(self, obs, map_tensor=None, inference=False):
        """
        前向传播

        Args:
            obs: 观测特征 [batch, FEATURE_DIM]
            map_tensor: 地图特征 (已包含在obs中，此参数为兼容性保留)
            inference: 是否推理模式

        Returns:
            logits: Q值分布的对数概率 [batch, ACTION_NUM, NUM_BINS]
            value: 期望Q值 [batch, 1]
        """
        batch_size = obs.size(0)

        # SimbaV2Critic输出分桶的logits
        # 输出形状: [batch, ACTION_NUM * NUM_BINS]
        value_bins = self.critic(obs)

        # reshape为 [batch, ACTION_NUM, NUM_BINS]
        logits = value_bins.view(batch_size, Config.ACTION_NUM, self.num_bins)

        # 计算期望Q值 (用于选择动作)
        probs = torch.softmax(logits, dim=-1)  # [batch, ACTION_NUM, NUM_BINS]
        q_values = torch.sum(probs * self.atoms.view(1, 1, -1), dim=-1)  # [batch, ACTION_NUM]

        # 取最大Q值作为状态价值估计 (DQN风格)
        value = q_values.max(dim=-1, keepdim=True)[0]  # [batch, 1]

        return logits, value

    def get_q_values(self, obs):
        """获取所有动作的Q值"""
        logits, _ = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        q_values = torch.sum(probs * self.atoms.view(1, 1, -1), dim=-1)
        return q_values

    def select_action(self, obs, epsilon=0.0, legal_action=None):
        """
        选择动作 (DQN风格)

        Args:
            obs: 观测
            epsilon: epsilon-greedy参数
            legal_action: 合法动作掩码

        Returns:
            action: 选择的动作索引
        """
        if np.random.random() < epsilon:
            # 随机探索
            if legal_action is not None:
                legal_indices = np.where(legal_action)[0]
                return np.random.choice(legal_indices)
            return np.random.randint(0, Config.ACTION_NUM)

        with torch.no_grad():
            q_values = self.get_q_values(obs)  # [1, ACTION_NUM]
            q_values = q_values.cpu().numpy()[0]

            # 应用合法动作掩码
            if legal_action is not None:
                q_values = q_values.copy()
                q_values[~legal_action] = -float('inf')

            return q_values.argmax()

    def l2_normalize(self):
        """应用L2归一化到所有HyperDense层"""
        l2normalize_network(self.critic)

    def get_initial_hidden(self, batch_size=1, device=None):
        """SimbaV2不需要隐藏状态，返回None"""
        return None

    def reset_hidden_state(self):
        """SimbaV2不需要重置隐藏状态"""
        pass

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
