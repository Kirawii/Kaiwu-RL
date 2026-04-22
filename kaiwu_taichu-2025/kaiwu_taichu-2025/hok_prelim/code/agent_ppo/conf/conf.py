#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np

# Configuration, including dimension settings, algorithm parameter settings.
# The last few configurations in the file are for the Kaiwu platform to use and should not be changed.
# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    GAMMA = 0.99

    # tdlambda
    TDLAMBDA = 0.95

    # Initial learning rate
    # 初始的学习率
    START_LR = 3e-4

    # entropy regularization coefficient
    # 熵正则化系数
    BETA_START = 1e-2

    # clip parameter
    # 裁剪参数
    CLIP_PARAM = 0.2

    # value function loss coefficient
    # 价值函数损失的系数
    VF_COEF = 0.5

    # actions
    # 动作
    ACTION_DIM = 1
    ACTION_NUM = 16

    # features
    # 特征
    HERO_FEATURE = [
        2,  # 当前英雄位置
        1,  # 闪现是否可用
        1,  # 闪现cd
        1,  # 加速buff是否存在
    ]
    HERO_FEATURE_DIM = sum(HERO_FEATURE)
    MAP_FEATURE_SHAPE = (4, 51, 51)
    MAP_FEATURE_DIM = int(np.prod(MAP_FEATURE_SHAPE))
    FEATURES = HERO_FEATURE + [
        MAP_FEATURE_DIM,  # 地图信息
    ]

    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    VALUE_NUM = 1
    DATA_SPLIT_SHAPE = [
        FEATURE_LEN,
        VALUE_NUM,
        VALUE_NUM,
        VALUE_NUM,
        VALUE_NUM,
        ACTION_DIM,
        ACTION_DIM,
        ACTION_NUM,
    ]
    data_len = sum(DATA_SPLIT_SHAPE)

    # Input dimension of reverb sample on learner. Note that different algorithms have different dimensions.
    # **Note**, this item must be configured correctly and should be aligned with the NumpyData2SampleData function data in definition.py
    # Otherwise the sample dimension error may be reported
    # learner上reverb样本的输入维度
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = data_len

    ############################################################
    # 在官方配置上新加入参数
    ############################################################
    SEQUENCE_LENGTH = 10  # 记忆长度

    #################### 环境 ####################
    TREASURE_NUM_PROB = [1] * 14  # 宝箱数均匀分布
    TREASURE_NUM_PROB = np.array(TREASURE_NUM_PROB) / np.sum(TREASURE_NUM_PROB)

    #################### 奖励 ####################
    REW_FINISH = 15  # 到终点奖励
    REW_TREASURE = 10  # 获得宝箱奖励, 到终点但错失的宝箱就是惩罚
    REW_FLASH = 0.1  # 闪现距离减15乘系数并对奖励做clip(-5, 0)范围
    REW_DISTANCE = 0.1  # 距离奖励向目标 (终点) 移动 (每一帧进行一次奖励)
    REW_HIT_WALL_PUNISH = 0.1  # 撞墙惩罚 (每一帧进行一次惩罚)
    REW_BUFF = 0.5  # 获得buff奖励
    REW_EACH_STEP_PUNISH = 0.02  # 每步的惩罚
    REW_MEMORY_PUNISH_COEF = np.array([  # 周围重复步数惩罚
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.8, 0.5, 0.0],
        [0.0, 0.8, 1.0, 0.8, 0.0],
        [0.0, 0.5, 0.8, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ], np.float32)
    REW_MEMORY_PUNISH_STEP = 2  # 周围重复步数惩罚的步数
    REW_GLOBAL_SCALE = 1.0  # 奖励缩放系数

if __name__ == '__main__':
    print(Config.FEATURE_LEN)
    print(Config.SAMPLE_DIM)
