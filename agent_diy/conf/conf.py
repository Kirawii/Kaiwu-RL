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

    # ==================== 特征维度配置 (完全复制agent_target_dqn) ====================
    # 英雄特征 (5D): pos_x, pos_z, flash_status, flash_cd_norm, buff_remain_time_norm
    HERO_FEATURE_DIM = 5

    # 地图特征形状 (4, 51, 51)
    MAP_FEATURE_SHAPE = (4, 51, 51)
    MAP_FEATURE_DIM = 4 * 51 * 51  # 10404

    # 总观测维度
    DIM_OF_OBSERVATION = HERO_FEATURE_DIM + MAP_FEATURE_DIM  # 10409

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

    # ==================== 完整密集奖励配置 (来自agent_target_dqn) ====================
    # 这些参数与agent_target_dqn完全一致，实现完整的10组件密集奖励

    # 1. 终点奖励
    REW_FINISH = 15.0

    # 2. 截断惩罚 (超时)
    REW_TRUNCATED_PUNISH = 80.0

    # 3. 宝箱奖励/惩罚
    REW_TREASURE = 10.0

    # 4. 闪现奖励/惩罚 (闪现距离-15后乘系数，并clip到-5,0)
    REW_FLASH = 0.1
    REW_FLASH_CLIP_MIN = -5.0
    REW_FLASH_CLIP_MAX = 0.0

    # 5. 撞墙惩罚
    REW_HIT_WALL_PUNISH = 0.1

    # 6. Buff奖励 (递减)
    REW_BUFF = 0.5
    REW_BUFF_DECAY = 0.5  # 每次buff奖励递减 (REW_BUFF * 0.5^buff_count)

    # 7. 每步惩罚
    REW_EACH_STEP_PUNISH = 0.02

    # 8. 距离奖励系数 (向目标移动的奖励)
    REW_DISTANCE = 0.1
    REW_DISTANCE_CLIP = 1.0  # 对未找到的目标裁剪距离变化到(-1,1)

    # 9. 周围重复步数惩罚
    REW_MEMORY_PUNISH_SIZE = 3  # 3x3区域
    REW_MEMORY_PUNISH_THRESHOLD = 9  # 阈值
    REW_MEMORY_PUNISH_COEF = 0.1

    # 10. 探索奖励 (王者无需，设为0)
    REW_EXPLORATION = 0.0

    # 11. 全局奖励缩放
    REW_GLOBAL_SCALE = 1.0

    # ==================== SimbaV2 网络配置 (来自agent_target_dqn) ====================
    # 参考亚军方案的超参数设置 (model.py中硬编码值)
    SIMBA_HIDDEN_DIM = 512
    SIMBA_NUM_BLOCKS = 2  # 2个LERP块
    # scaler_init = sqrt(2 / hidden_dim) = sqrt(2/512) ≈ 0.0625
    SIMBA_SCALER_INIT = 0.0625
    # scaler_scale = sqrt(2 / hidden_dim) = 0.0625 (亚军方案与init相同)
    SIMBA_SCALER_SCALE = 0.0625
    # alpha_init = 1 / (num_blocks + 1) = 1/3 ≈ 0.333
    SIMBA_ALPHA_INIT = 0.3333
    # alpha_scale = 1 / sqrt(hidden_dim) = 1/sqrt(512) ≈ 0.0442
    SIMBA_ALPHA_SCALE = 0.0442
    SIMBA_C_SHIFT = 3.0
    # num_bins = action_shape = 16 (输出16个Q值)
    SIMBA_NUM_BINS = 16

    # ==================== DQN 配置 ====================
    GAMMA = 0.995  # 折扣因子
    TARGET_UPDATE_FREQ = 200  # Target网络更新频率
    EPSILON_MIN = 0.1
    EPSILON_MAX = 1.0
    EPSILON_DECAY = 1e-6
    START_LR = 1e-4  # 初始学习率

    # Reverb样本维度 (与agent_target_dqn一致)
    # SAMPLE_DIM = 2 * (DIM_OF_OBSERVATION + ACTION_NUM) + 4
    # = 2 * (10409 + 16) + 4 = 20854
    SAMPLE_DIM = 20854

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
