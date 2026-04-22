#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
SimbaV2 网络层实现
"""

import torch
import torch.nn as nn
import math
from agent_diy.model.simbaV2.common.math import l2normalize


class HyperDense(nn.Module):
    """超密集层，带L2归一化"""

    def __init__(self, in_dim, out_dim, scaler_init=1.0, scaler_scale=1.0, c_shift=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scaler_init = scaler_init
        self.scaler_scale = scaler_scale
        self.c_shift = c_shift

        # 权重初始化
        self.hyper_dense = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.hyper_dense.weight, gain=scaler_init)

        # 缩放参数
        self.log_scaler = nn.Parameter(torch.zeros(out_dim))
        nn.init.constant_(self.log_scaler, math.log(scaler_scale))

    def forward(self, x):
        # L2归一化权重
        weight = l2normalize(self.hyper_dense.weight, dim=0)
        # 应用缩放
        scaler = torch.exp(self.log_scaler)
        # 线性变换
        out = torch.nn.functional.linear(x, weight, None)
        out = out * scaler
        return out


class HyperEmbedder(nn.Module):
    """超嵌入层"""

    def __init__(self, in_dim, hidden_dim, scaler_init, scaler_scale, c_shift):
        super().__init__()
        self.embed = HyperDense(
            in_dim, hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift
        )

    def forward(self, x):
        return torch.relu(self.embed(x))


class HyperLERPBlock(nn.Module):
    """超LERP残差块 (Linear Interpolation Block)"""

    def __init__(self, in_dim, hidden_dim, scaler_init, scaler_scale, alpha_init, alpha_scale):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # 两个HyperDense层
        self.dense1 = HyperDense(
            in_dim, hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale
        )
        self.dense2 = HyperDense(
            hidden_dim, in_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale
        )

        # LERP插值参数
        self.log_alpha = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.log_alpha, math.log(alpha_init))
        self.alpha_scale = alpha_scale

    def forward(self, x):
        residual = x

        # 前向通过两个dense层
        h = torch.relu(self.dense1(x))
        h = self.dense2(h)

        # LERP插值: output = alpha * h + (1 - alpha) * residual
        alpha = torch.exp(self.log_alpha) * self.alpha_scale
        alpha = torch.sigmoid(alpha)  # 限制在0-1

        out = alpha * h + (1 - alpha) * residual
        return torch.relu(out)


class HyperDiscreteValueHead(nn.Module):
    """离散值头 (用于DQN输出Q值)"""

    def __init__(self, in_dim, hidden_dim, num_bins, scaler_init, scaler_scale):
        super().__init__()
        self.predictor = HyperDense(
            in_dim, num_bins,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale
        )

    def forward(self, x):
        return self.predictor(x)


class HyperNormalPolicyHead(nn.Module):
    """正态策略头 (用于连续动作)"""

    def __init__(self, in_dim, hidden_dim, action_dim, scaler_init, scaler_scale):
        super().__init__()
        self.mean_layer = HyperDense(
            in_dim, action_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale
        )
        self.log_std_layer = HyperDense(
            in_dim, action_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale
        )

    def forward(self, x):
        raw_mean = self.mean_layer(x)
        raw_log_std = self.log_std_layer(x)
        return raw_mean, raw_log_std
