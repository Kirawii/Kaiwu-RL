#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
SimbaV2 网络架构实现
参考 kaiwu_taichu-2025 亚军方案
"""

import torch
import torch.nn as nn
import math
from agent_diy.model.simbaV2.agents.layers import (
    HyperDense,
    HyperEmbedder,
    HyperLERPBlock,
    HyperDiscreteValueHead,
)
from agent_diy.model.simbaV2.common.math import l2normalize


class SimbaV2Critic(nn.Module):
    """
    SimbaV2 Critic网络 (用于DQN)
    架构: Embedder -> Encoder (2 LERP blocks) -> Predictor
    """

    def __init__(
        self,
        in_dim,
        num_blocks,
        hidden_dim,
        scaler_init,
        scaler_scale,
        alpha_init,
        alpha_scale,
        c_shift,
        num_bins,
    ):
        super().__init__()
        self.embedder = HyperEmbedder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
        )
        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    in_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
                for _ in range(num_blocks)
            ]
        )
        self.predictor = HyperDiscreteValueHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, x):
        y = self.embedder(x)
        z = self.encoder(y)
        value_bins = self.predictor(z)
        return value_bins


@torch.no_grad()
def l2normalize_network(network):
    """Apply L2 normalization to all hyper-dense layers in the network"""

    def norm(m):
        if isinstance(m, HyperDense):
            assert m.hyper_dense.weight.ndim == 2
            m.hyper_dense.weight.set_(l2normalize(m.hyper_dense.weight, dim=0))

    network.apply(norm)
