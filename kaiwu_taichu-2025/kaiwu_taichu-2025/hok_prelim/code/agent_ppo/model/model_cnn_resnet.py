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

        # Main encode network
        self.encoder = nn.Sequential(  # (5, 128, 128) -> (256,)
            make_cnn_layer(5, 32, kernel_size=7, stride=4, padding=2),  # (32, 32, 32)
            nn.ReLU(),
            make_cnn_layer(32, 64, kernel_size=3, stride=2, padding=1),   # (64, 16, 16)
            nn.ReLU(),
            # ResidualBlock((64, 16, 16), type='cnn'),
            make_cnn_layer(64, 128, kernel_size=3, stride=2, padding=1),   # (128, 8, 8)
            nn.ReLU(),
            # ResidualBlock((128, 8, 8), type='cnn'),
            # ResidualBlock((128, 8, 8), type='cnn'),
            nn.AvgPool2d(kernel_size=2, stride=2),  # (128, 2, 2)
            nn.Flatten(),
            MLP([128 * 4 * 4, 512, 256], "encoder_mlp"),  # (512,)
        )

        self.action_mlp = MLP([256 + 5, 64, self.action_num], "action_mlp")
        self.value_mlp = MLP([256 + 5, 64, self.value_num], "value_mlp")

    def process_legal_action(self, action, legal_action):
        action_max, _ = torch.max(action * legal_action, 1, True)
        action = action - action_max  # 确保数值稳定性
        action = action * legal_action  # mask有效动作
        action = action + 1e5 * (legal_action - 1)  # 无效动作减到-1e5以下
        return action

    def forward(self, feature, legal_action):
        x, map = feature[:, :Config.HERO_FEATURE_DIM], feature[:, Config.HERO_FEATURE_DIM:].reshape(-1, *Config.MAP_FEATURE_SHAPE)
        # 主encoder处理
        feature = torch.cat([x, self.encoder(map)], 1)

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
    fc_layer = nn.Linear(in_features, out_features)
    if init_method == 'orthogonal':
        nn.init.orthogonal_(fc_layer.weight)
    elif init_method == 'kaiming':
        nn.init.kaiming_uniform_(fc_layer.weight, nonlinearity='relu')
    nn.init.zeros_(fc_layer.bias)
    return fc_layer

def make_cnn_layer(
        in_channels: int, out_channels: int, init_method: str = 'orthogonal',
        kernel_size: int = 3, stride: int = 1, padding: int = 1
    ):
    cnn_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    if init_method == 'orthogonal':
        nn.init.orthogonal_(cnn_layer.weight)
    elif init_method == 'kaiming':
        nn.init.kaiming_uniform_(cnn_layer.weight, nonlinearity='relu')
    nn.init.zeros_(cnn_layer.bias)

    return cnn_layer

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
    def __init__(self, dim: int | tuple, type='linear'):
        super().__init__()
        if type == 'linear':
            self.net = make_fc_layer(dim, dim * 4, init_method='kaiming')
            self.bottle_net = make_fc_layer(dim * 4, dim, init_method='kaiming')
        elif type == 'cnn':  # dim=(C, H, W)
            self.net = make_cnn_layer(dim[0], dim[0], init_method='kaiming')
            self.bottle_net = make_cnn_layer(dim[0], dim[0], init_method='kaiming')
        self.layer_norm = nn.LayerNorm(np.prod(dim))

    def forward(self, x):
        residual = x
        shape = x.shape
        x = self.layer_norm(x.reshape(shape[0], -1))
        x = x.reshape(shape)
        x = self.net(x)
        x = torch.relu(x)
        x = self.bottle_net(x)
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
