#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

峡谷追猎 - DIY Agent 配置
包含完整特征：英雄 + 2怪物 + 10宝箱 + 2Buff + 地图
优化版本：参考夺冠经验调参
"""


class Config:
    """配置类 - 包含维度设置、算法参数设置"""

    # ==================== 特征维度配置 ====================
    # 英雄自身特征 (6D): pos_x, pos_z, flash_cd, flash_ready, buff_duration, speed_boost
    HERO_FEATURE_DIM = 6

    # 怪物特征 (6D x 2): is_in_view, pos_x, pos_z, speed, distance_bucket, relative_dir
    MONSTER_FEATURE_DIM = 6
    MONSTER_COUNT = 2

    # 宝箱特征 (6D x 10): is_active, pos_x, pos_z, distance_bucket, relative_dir, priority
    TREASURE_FEATURE_DIM = 6
    TREASURE_COUNT = 10

    # Buff特征 (6D x 2): is_active, pos_x, pos_z, distance_bucket, relative_dir, remaining_time
    BUFF_FEATURE_DIM = 6
    BUFF_COUNT = 2

    # 地图特征 (25D): 以英雄为中心的 5x5 局部地图通行性
    MAP_LOCAL_DIM = 25

    # 进度特征 (4D): step_norm, survival_ratio, monster2_timer, danger_level
    PROGRESS_DIM = 4

    # 总特征维度 (包含合法动作掩码作为输入特征)
    # 6 + 12 + 60 + 12 + 25 + 4 + 16 = 135
    FEATURE_DIM = 135

    # ==================== CNN地图配置 (参考亚军) ====================
    # 4通道51x51局部地图 (裁剪自128x128全局地图)
    MAP_CHANNELS = 4
    MAP_SIZE_CNN = 51  # 局部输入尺寸
    MAP_SIZE_GLOBAL = 128  # 全局地图尺寸
    MAP_TENSOR_SHAPE = (MAP_CHANNELS, MAP_SIZE_CNN, MAP_SIZE_CNN)  # 4x51x51 = 10404

    # 通道定义 (参考亚军: 0=障碍, 1=记忆, 2=Buff/宝箱, 3=终点)
    # 通道2: -1=Buff精确, -0.5=Buff大致, 0=无, 0.5=宝箱大致, 1=宝箱精确

    # ==================== 动作空间配置 ====================
    # 16维动作: 0-7 移动, 8-15 闪现
    ACTION_NUM = 16
    MOVE_ACTION_NUM = 8
    FLASH_ACTION_NUM = 8

    # ==================== 价值头配置 ====================
    # 单头价值 (与基线一致)
    VALUE_NUM = 1

    # ==================== PPO 超参数 (优化后) ====================
    # 折扣因子 - readme建议: 0.995 -> 0.997 -> 0.9975 逐步增加
    GAMMA_START = 0.995
    GAMMA_END = 0.9975
    GAMMA_DECAY_STEPS = 100000  # 在10万步内逐渐增加到最大值

    # GAE lambda - readme建议: 0.95 -> 0.97 -> 0.98 逐步增加
    LAMDA_START = 0.95
    LAMDA_END = 0.98
    LAMDA_DECAY_STEPS = 100000

    # 学习率
    INIT_LEARNING_RATE_START = 3e-4
    LR_DECAY = True
    LR_DECAY_RATE = 0.999
    LR_MIN = 3e-6  # readme建议最小3e-6，再小精度不够

    # PPO clip参数
    CLIP_PARAM = 0.2

    # 价值损失系数 - readme建议 policy:value = 2:1，所以 vf_coef = 0.5
    VF_COEF = 0.5

    # 熵正则化系数 - readme建议 0.01-0.015
    BETA_START = 0.015
    BETA_END = 0.001  # 逐渐衰减
    BETA_DECAY_STEPS = 50000

    # 梯度裁剪
    GRAD_CLIP_RANGE = 0.5

    # ==================== 动作掩码损失权重 (新增) ====================
    ACTION_MASK_LOSS_COEF = 0.1  # 动作约束损失系数
    FLASH_WASTE_PENALTY = 0.5    # 浪费闪现的惩罚
    NO_FLASH_WHEN_DANGER_PENALTY = 1.0  # 危险时不闪现的惩罚

    # ==================== 课程学习配置 (新增) ====================
    CURRICULUM_ENABLED = True
    EARLY_PHASE_RATIO = 0.5      # 前期占比50%，更多时间鼓励收集
    LATE_PHASE_RATIO = 0.2       # 后期占比20%，减少保守阶段

    # ==================== 奖励权重 (情景奖励优化) ====================
    # 基础奖励
    SURVIVE_REWARD_BASE = 0.01

    # 宝箱收集奖励 - 前期重收集，后期重生存
    TREASURE_REWARD_EARLY = 15.0
    TREASURE_REWARD_LATE = 5.0

    # 距离塑形奖励系数 (远离怪物)
    DISTANCE_SHAPING_COEF = 0.05

    # 探索奖励
    EXPLORATION_REWARD = 0.001

    # 闪现使用奖励 (参考readme的精准释放思想)
    FLASH_ESCAPE_REWARD = 2.0        # 成功逃脱大奖励
    FLASH_WASTE_PENALTY_REWARD = -1.0  # 浪费闪现惩罚

    # 情景奖励 (核心)
    DANGER_TREASURE_PENALTY = -3.0   # 危险时冒险收集宝箱惩罚
    SAFE_NO_EXPLORATION_PENALTY = -0.5  # 安全时不探索惩罚 (加重)
    PROXIMITY_ESCAPE_REWARD = 1.0    # 近距离成功逃脱奖励

    # 终局奖励
    DEATH_PENALTY = -10.0
    WIN_REWARD = 10.0

    # 步数奖励 - 前期鼓励探索，后期鼓励生存
    STEP_REWARD_COEF = 0.008  # 降低基础步数奖励，避免过于保守

    # ==================== 新增奖励配置 (参考优秀经验) ====================
    # Buff奖励 (高奖励鼓励收集)
    REW_BUFF = 8.0  # 第一个Buff高奖励
    REW_BUFF_DECAY = 0.5  # 第二个Buff奖励 8 * 0.5 = 4

    # 闪现奖励/惩罚
    REW_FLASH_DISTANCE_COEF = 0.1
    REW_FLASH_CLIP_MIN = -5.0
    REW_FLASH_CLIP_MAX = 0.0

    # 撞墙惩罚
    REW_HIT_WALL_PUNISH = 0.1

    # 距离奖励 (优先最近宝箱和Buff)
    REW_DISTANCE = 0.3  # 增加距离引导强度
    REW_DISTANCE_CLIP = 2.0  # 允许更大的距离变化奖励

    # 周围重复步数惩罚
    REW_MEMORY_PUNISH_SIZE = 3  # 3x3区域
    REW_MEMORY_PUNISH_THRESHOLD = 9  # 阈值
    REW_MEMORY_PUNISH_COEF = 0.1

    # 探索奖励
    REW_EXPLORATION = 0.001

    # ==================== 网络结构配置 (FC+GRU混合) ====================
    # FC层维度
    FC_HIDDEN_DIM = 192
    FC_LAYER_NUM = 3  # 3层FC

    # GRU层维度 - readme建议 FC:GRU = 3:1
    GRU_HIDDEN_DIM = 64
    GRU_LAYER_NUM = 1

    # 中间层
    MID_DIM = 128

    # ==================== 地图常量 ====================
    MAP_SIZE = 128.0
    MAX_DIST_BUCKET = 5.0
    MAX_FLASH_CD = 2000.0
    MAX_BUFF_DURATION = 50.0
    MAX_MONSTER_SPEED = 5.0
    VIEW_RANGE = 10

    # ==================== 危险等级阈值 ====================
    DANGER_THRESHOLD_HIGH = 0.7   # 高危
    DANGER_THRESHOLD_MID = 0.4    # 中危
    DANGER_THRESHOLD_LOW = 0.2    # 低危

    # ==================== 闪现冷却阈值 ====================
    FLASH_CD_THRESHOLD = 10  # 小于10步认为"即将可用"
