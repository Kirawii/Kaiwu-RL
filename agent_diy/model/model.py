#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

峡谷追猎 - DIY Agent DQN 网络模型
完全参考 agent_target_dqn 亚军方案
架构: CNN编码器 + SimbaV2Critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from agent_diy.conf.conf import Config
from agent_diy.model.simbaV2.agents.networks import SimbaV2Critic, l2normalize_network


class Model(nn.Module):
    """
    DQN 网络模型
    输入: 英雄特征 + 地图特征
    处理: CNN编码地图 -> 拼接英雄特征 -> SimbaV2Critic -> Q值
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_diy_dqn_simbaV2"
        self.device = device

        # 特征维度
        self.hero_feature_dim = Config.HERO_FEATURE_DIM  # 5 (位置2D + 闪现可用1D + 闪现cd 1D + buff 1D)
        self.map_shape = Config.MAP_TENSOR_SHAPE  # (4, 51, 51)

        # CNN地图编码器
        # 输入: 4x51x51 -> 输出: 512维特征
        self.q_cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            # 计算输出维度: 51->(51-7)/2+1=23->(23-5)/2+1=10->(23-3)/2+1 不对，重新算
            # 51 - 7 + 1 = 45, /2 = 22 (向下取整) + 1 = 23? 不对
            # Conv2d stride=2: (W - K + 2P) / S + 1, 默认padding=0
            # L1: (51 - 7) / 2 + 1 = 22 + 1 = 23
            # L2: (23 - 5) / 2 + 1 = 9 + 1 = 10
            # L3: (10 - 3) / 1 + 1 = 8
            # 输出: 64 * 8 * 8 = 4096
            MLP([4096, 512], "q_cnn", non_linearity_last=True)
        )

        # SimbaV2 Critic
        # 输入: 512(CNN输出) + 5(英雄特征) = 517
        # 输出: ACTION_NUM (16个Q值)
        simba_hidden_dim = 512
        simba_num_block = 2
        self.simba_critic = SimbaV2Critic(
            in_dim=512 + self.hero_feature_dim,
            num_blocks=simba_num_block,
            hidden_dim=simba_hidden_dim,
            scaler_init=math.sqrt(2 / simba_hidden_dim),
            scaler_scale=math.sqrt(2 / simba_hidden_dim),
            alpha_init=1 / (simba_num_block + 1),
            alpha_scale=1 / math.sqrt(simba_hidden_dim),
            c_shift=3.0,
            num_bins=Config.ACTION_NUM,  # 输出16个Q值
        )

        # 初始化L2归一化
        l2normalize_network(self.simba_critic)

    def forward(self, feature):
        """
        前向传播

        Args:
            feature: [batch, DIM_OF_OBSERVATION] 完整特征向量
                    其中前HERO_FEATURE_DIM维是英雄特征，其余是地图特征

        Returns:
            logits: [batch, ACTION_NUM] Q值
        """
        # 分离英雄特征和地图特征
        # 输入格式: [hero_feature(5D), map_feature(4*51*51=10404D)]
        x = feature[:, :self.hero_feature_dim]
        map_flat = feature[:, self.hero_feature_dim:]

        # reshape地图特征: [batch, 4, 51, 51]
        map_tensor = map_flat.reshape(-1, *self.map_shape)

        # CNN编码地图
        map_encoded = self.q_cnn(map_tensor)

        # 拼接英雄特征和地图编码
        combined = torch.cat([map_encoded, x], dim=1)

        # SimbaV2输出Q值 (加微小噪声防止零向量)
        logits = self.simba_critic(combined + 1e-8)

        return logits

    def get_q_values(self, feature):
        """获取所有动作的Q值"""
        return self.forward(feature)

    def l2_normalize(self):
        """应用L2归一化"""
        l2normalize_network(self.simba_critic)

    def get_initial_hidden(self, batch_size=1, device=None):
        """DQN不需要隐藏状态"""
        return None

    def reset_hidden_state(self):
        """DQN不需要重置隐藏状态"""
        pass

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()


class MLP(nn.Module):
    """多层感知机"""

    def __init__(
        self,
        fc_feat_dim_list,
        name,
        non_linearity=nn.ReLU,
        non_linearity_last=False,
    ):
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module(f"{name}_fc{i + 1}", fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(f"{name}_relu{i + 1}", non_linearity())

    def forward(self, data):
        return self.fc_layers(data)


def make_fc_layer(in_features, out_features, init_method='orthogonal'):
    """创建全连接层"""
    fc_layer = nn.Linear(in_features, out_features)
    if init_method == 'orthogonal':
        nn.init.orthogonal_(fc_layer.weight)
    elif init_method == 'kaiming':
        nn.init.kaiming_uniform_(fc_layer.weight, nonlinearity='relu')
    nn.init.zeros_(fc_layer.bias)
    return fc_layer
