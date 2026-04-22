#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

峡谷追猎 - DIY Agent 神经网络模型 v3
升级：CNN地图处理 + 空间注意力 + FC+GRU混合
参考：多通道地图表示经验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agent_diy.conf.conf import Config


def make_fc_layer(in_features, out_features, gain=np.sqrt(2)):
    """创建正交初始化的全连接层"""
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=gain)
    nn.init.zeros_(fc.bias.data)
    return fc


class MapEncoder(nn.Module):
    """
    地图CNN编码器
    输入: 4通道51x51地图
        - 通道0: 障碍物 (-1=未知, 0=障碍, 1=通路)
        - 通道1: 访问记忆 (访问次数归一化)
        - 通道2: 宝箱 (0.5=大致位置, 1.0=准确位置)
        - 通道3: Buff (-0.5=大致, -1.0=准确, 负数区分类型)
    输出: 地图特征向量
    """
    def __init__(self, map_size=51, out_dim=128):
        super().__init__()
        self.map_size = map_size

        # 卷积层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 计算展平后维度
        # 经过3层卷积，尺寸不变(有padding)
        self.flat_dim = 64 * map_size * map_size

        # 降维到指定输出维度
        self.fc = make_fc_layer(self.flat_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, map_tensor):
        """
        Args:
            map_tensor: [batch, 4, 51, 51]
        Returns:
            map_feature: [batch, out_dim]
            attn_map: [batch, 1, 51, 51] (可视化用)
        """
        x = self.activation(self.conv1(map_tensor))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))

        # 空间注意力
        attn = self.spatial_attn(x)
        x = x * attn  # 加权

        # 展平并降维
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc(x))

        return x, attn


class PSCNModule(nn.Module):
    """PSCN (Power Series Connected Neural Network) 模块"""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = make_fc_layer(dim, dim // 2)
        self.fc2 = make_fc_layer(dim // 2, dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return self.activation(x + residual)


class Model(nn.Module):
    """
    Actor-Critic 网络模型 v3
    - CNN地图编码
    - 空间注意力
    - FC+GRU混合
    - 辅助任务：危险预测、地图重建
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_diy_v3"
        self.device = device

        # ==================== 地图CNN编码器 ====================
        self.map_encoder = MapEncoder(map_size=51, out_dim=128)

        # ==================== 非地图特征处理 ====================
        # 英雄 + 怪物 + 宝箱列表 + Buff列表 + 地图(25D) + 进度 + 合法动作掩码
        non_map_dim = (
            Config.HERO_FEATURE_DIM +
            Config.MONSTER_FEATURE_DIM * Config.MONSTER_COUNT +
            Config.TREASURE_FEATURE_DIM * Config.TREASURE_COUNT +
            Config.BUFF_FEATURE_DIM * Config.BUFF_COUNT +
            Config.MAP_LOCAL_DIM +  # 25D 地图特征
            Config.PROGRESS_DIM +
            Config.ACTION_NUM  # legal_action
        )

        self.non_map_fc = nn.Sequential(
            make_fc_layer(non_map_dim, 128),
            nn.ReLU(),
        )

        # ==================== 特征融合 ====================
        # 地图特征128D + 非地图特征128D = 256D
        fusion_dim = 256

        # ==================== FC骨干 (3层，占比3/4) ====================
        self.fc_backbone = nn.Sequential(
            make_fc_layer(fusion_dim, 192),
            nn.ReLU(),
            PSCNModule(192),
            make_fc_layer(192, 192),
            nn.ReLU(),
        )

        # ==================== GRU层 (1层，占比1/4) ====================
        self.gru = nn.GRU(
            input_size=192,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # ==================== 特征融合层 ====================
        # 拼接FC输出(192D)和GRU输出(64D)
        self.fusion = nn.Sequential(
            make_fc_layer(256, 128),
            nn.ReLU(),
        )

        # ==================== Actor 头 ====================
        self.actor_hidden = nn.Sequential(
            make_fc_layer(128, 64),
            nn.ReLU(),
        )
        self.actor_head = make_fc_layer(64, Config.ACTION_NUM, gain=0.01)

        # ==================== Critic 头 ====================
        self.critic_hidden = nn.Sequential(
            make_fc_layer(128, 64),
            nn.ReLU(),
        )
        self.critic_head = make_fc_layer(64, 1, gain=0.01)

        # ==================== 辅助任务 ====================
        # 1. 危险等级预测
        self.danger_predictor = nn.Sequential(
            make_fc_layer(128, 32),
            nn.ReLU(),
            make_fc_layer(32, 1, gain=0.01),
            nn.Sigmoid(),
        )

        # 2. 地图重建 (辅助学习地图表示)
        self.map_decoder = nn.Sequential(
            make_fc_layer(128, 256),
            nn.ReLU(),
            make_fc_layer(256, 4 * 51 * 51),  # 重建4通道51x51
        )

        # 隐藏状态
        self.hidden_state = None

    def forward(self, obs, map_tensor=None, inference=False, hidden_state=None):
        """
        前向传播

        Args:
            obs: 非地图特征 [batch, non_map_dim]
            map_tensor: 地图特征 [batch, 4, 51, 51] (可选)
            inference: 是否推理模式
            hidden_state: GRU隐藏状态
        """
        batch_size = obs.size(0)

        # 处理地图特征
        if map_tensor is not None:
            map_feature, attn_map = self.map_encoder(map_tensor)
        else:
            # 如果没有地图输入，使用零向量
            map_feature = torch.zeros(batch_size, 128, device=obs.device)
            attn_map = None

        # 处理非地图特征
        non_map_feature = self.non_map_fc(obs)

        # 特征融合
        combined = torch.cat([map_feature, non_map_feature], dim=-1)

        # FC骨干
        fc_features = self.fc_backbone(combined)

        # GRU
        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, 64, device=obs.device)

        gru_input = fc_features.unsqueeze(1)
        gru_output, new_hidden_state = self.gru(gru_input, hidden_state)
        gru_features = gru_output.squeeze(1)

        # 最终融合
        fused = torch.cat([fc_features, gru_features], dim=-1)
        hidden = self.fusion(fused)

        # Actor
        actor_hidden = self.actor_hidden(hidden)
        logits = self.actor_head(actor_hidden)

        # Critic
        critic_hidden = self.critic_hidden(hidden)
        value = self.critic_head(critic_hidden)

        # 辅助任务
        danger_pred = self.danger_predictor(hidden)

        # 地图重建 (训练时使用)
        if not inference:
            map_recon = self.map_decoder(map_feature)
            map_recon = map_recon.view(batch_size, 4, 51, 51)
        else:
            map_recon = None

        # 更新隐藏状态
        if inference and batch_size == 1:
            self.hidden_state = new_hidden_state.detach()

        return logits, value, new_hidden_state, danger_pred, attn_map, map_recon

    def get_initial_hidden(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, 64, device=device)

    def reset_hidden_state(self):
        self.hidden_state = None

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
