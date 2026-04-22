#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from typing import List
import torch
from torch import nn
import numpy as np
from agent_ppo.conf.conf import Config

import sys
import os

if os.path.basename(sys.argv[0]) == "learner.py":
    import torch

    torch.set_num_interop_threads(2)
    torch.set_num_threads(2)
else:
    import torch

    torch.set_num_interop_threads(4)
    torch.set_num_threads(4)


class NetworkModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        # feature configure parameter
        # 特征配置参数
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.feature_split_shape = Config.FEATURE_SPLIT_SHAPE
        self.action_num = Config.ACTION_NUM
        self.feature_len = Config.FEATURE_LEN
        self.value_num = Config.VALUE_NUM

        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF

        self.clip_param = Config.CLIP_PARAM

        self.data_len = Config.data_len

        # 预编码
        # 所有position特征共享一个编码器
        self.position_encoder = MLP([Config.POSITION_FEATURE_DIM, 16, 8], 'position_encoder', non_linearity_last=True)
        # 地图信息
        self.map_encoder = MLP([Config.MAP_FEATURE_DIM, 128, 64], 'map_encoder', non_linearity_last=True)
        self.memory_encoder = MLP([Config.MEMORY_FEATRUE_DIM, 128, 64], 'memory_encoder', non_linearity_last=True)

        # Main encode network
        self.encoder = nn.Sequential(
            MLP([
                sum(Config.HERO_FEATURE) + Config.ACTION_NUM +
                8 * Config.POSITION_FEATURE_NUM +
                64 + 64,  # 277
                256
            ], "first_mlp", non_linearity_last=True),
            ResidualBlock(256, 256),
        )

        self.action_mlp = MLP([256, 64, self.action_num], "action_mlp")
        self.value_mlp = MLP([256, 64, self.value_num], "value_mlp")

    def process_legal_action(self, action, legal_action):
        action_max, _ = torch.max(action * legal_action, 1, True)
        action = action - action_max  # 确保数值稳定性
        action = action * legal_action  # mask有效动作
        action = action + 1e5 * (legal_action - 1)  # 无效动作减到-1e5以下
        return action
    
    def encode(self, feature):
        hero, action_mask, *positions, map, memory = torch.split(
            feature, [
                sum(Config.HERO_FEATURE),
                Config.ACTION_NUM,
                *([Config.POSITION_FEATURE_DIM] * Config.POSITION_FEATURE_NUM),
                Config.MAP_FEATURE_DIM,
                Config.MEMORY_FEATRUE_DIM
            ], dim=-1
        )
        x = [hero, action_mask]
        for position in positions:
            x.append(self.position_encoder(position))
        x.append(self.map_encoder(map))
        x.append(self.memory_encoder(memory))
        x = torch.cat(x, axis=-1)
        x = self.encoder(x)
        return x

    def forward(self, feature, legal_action):
        # 主encoder处理
        feature = self.encode(feature)

        # 处理动作和值
        action_mlp_out = self.action_mlp(feature)
        action_out = self.process_legal_action(action_mlp_out, legal_action)

        prob = torch.nn.functional.softmax(action_out, dim=1)
        value = self.value_mlp(feature)

        return prob, value


class NetworkModelActor(NetworkModelBase):  # 在agent中使用, 传入为单个数据
    def format_data(self, obs, legal_action):
        return (
            torch.tensor(obs).to(torch.float32),
            torch.tensor(legal_action).to(torch.float32),
        )


class NetworkModelLearner(NetworkModelBase):  # 在algorithm中learn中使用, 传入为list
    def format_data(self, datas):
        return datas.view(-1, self.data_len).float().split(self.data_split_shape, dim=1)

    def forward(self, data_list, inference=False):
        feature = data_list[0]
        legal_action = data_list[-1]
        return super().forward(feature, legal_action)

def make_fc_layer(in_features: int, out_features: int, init_method: str = 'orthogonal'):
    # Wrapper function to create and initialize a linear layer
    # 创建并初始化一个线性层
    fc_layer = nn.Linear(in_features, out_features)

    # initialize weight and bias
    # 初始化权重及偏移量
    if init_method == 'orthogonal':
        nn.init.orthogonal_(fc_layer.weight)
    elif init_method == 'kaiming':
        nn.init.kaiming_uniform_(fc_layer.weight, nonlinearity='relu')
    nn.init.zeros_(fc_layer.bias)

    return fc_layer

class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        # Create a MLP object
        # 创建一个 MLP 对象
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            # no relu for the last fc layer of the mlp unless required
            # 除非有需要，否则 mlp 的最后一个 fc 层不使用 relu
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)

class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = make_fc_layer(in_features, out_features * 4, init_method='kaiming')
        self.bottle_linear = make_fc_layer(out_features * 4, out_features, init_method='kaiming')
        self.layer_norm = nn.LayerNorm(in_features)
    
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear(x)
        x = torch.relu(x)
        x = self.bottle_linear(x)
        return residual + x

if __name__ == '__main__':
    # Example usage
    model = NetworkModelActor()
    print(model)
    obs = np.random.rand(10, Config.FEATURE_LEN).astype(np.float32)  # Example observation
    legal_action = np.random.randint(0, 2, (10, Config.ACTION_NUM)).astype(np.float32)  # Example legal actions
    formatted_data = model.format_data(obs, legal_action)
    output = model(*formatted_data)
    print(output)
    from agent_ppo.utils import PATH_DEBUG
    torch.save(model.state_dict(), PATH_DEBUG / 'model.pth')
