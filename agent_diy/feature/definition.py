#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

峡谷追猎 - DIY Agent 特征定义与样本处理
优化版本：课程学习 + 情景奖励 + GAE + 动态参数
"""

import numpy as np
try:
    from common_python.utils.common_func import create_cls
except ImportError:
    # 本地开发环境 - 简单实现
    from collections import namedtuple
    def create_cls(name, **kwargs):
        fields = list(kwargs.keys())
        return namedtuple(name, fields, defaults=[None]*len(fields))
from agent_diy.conf.conf import Config


# ==================== 数据结构定义 ====================

ObsData = create_cls(
    "ObsData",
    feature=None,        # 非地图特征向量
    legal_act=None,      # 合法动作掩码 (16维)
    map_tensor=None,     # 4通道51x51地图张量
)


ActData = create_cls(
    "ActData",
    action=None,         # 动作索引 (0-15)
    d_action=None,       # 贪心动作 (DQN中与action相同)
    prob=None,           # 动作概率分布 (DQN中不使用)
    value=None,          # 状态价值
    move_dir=None,       # 移动方向 (0-7, 来自agent_target_dqn)
    use_talent=None,     # 是否使用闪现 (0或1, 来自agent_target_dqn)
)


SampleData = create_cls(
    "SampleData",
    obs=None,           # 观测特征 [DIM_OF_OBSERVATION]
    _obs=None,          # 下一状态观测 [DIM_OF_OBSERVATION]
    obs_legal=None,     # 合法动作掩码 [ACTION_NUM]
    _obs_legal=None,    # 下一状态合法动作掩码 [ACTION_NUM]
    act=None,           # 执行动作 (int)
    rew=None,           # 即时奖励 (float)
    ret=None,           # 返回值 (float, 备用)
    done=None,          # 是否结束 (float)
)


# ==================== 动态参数获取 (课程学习) ====================

def get_current_gamma(train_step):
    """获取当前的gamma值 (渐进增加)"""
    if not hasattr(Config, 'GAMMA_DECAY_STEPS'):
        return Config.GAMMA_START

    ratio = min(train_step / Config.GAMMA_DECAY_STEPS, 1.0)
    return Config.GAMMA_START + (Config.GAMMA_END - Config.GAMMA_START) * ratio


def get_current_lambda(train_step):
    """获取当前的lambda值 (渐进增加)"""
    if not hasattr(Config, 'LAMDA_DECAY_STEPS'):
        return Config.LAMDA_START

    ratio = min(train_step / Config.LAMDA_DECAY_STEPS, 1.0)
    return Config.LAMDA_START + (Config.LAMDA_END - Config.LAMDA_START) * ratio


def get_current_beta(train_step):
    """获取当前的熵系数 (渐进衰减)"""
    if not hasattr(Config, 'BETA_DECAY_STEPS'):
        return Config.BETA_START

    ratio = min(train_step / Config.BETA_DECAY_STEPS, 1.0)
    return Config.BETA_START + (Config.BETA_END - Config.BETA_START) * ratio


# ==================== 情景奖励 (核心优化) ====================

def calculate_situational_rewards(cur_info, next_info):
    """
    计算情景奖励 - 参考夺冠经验

    关键情景:
    1. 危险时冒险收集宝箱 -> 惩罚
    2. 安全时不探索 -> 轻微惩罚
    3. 近距离成功逃脱 -> 大奖励
    4. 浪费闪现 -> 惩罚
    5. 危险时使用闪现逃脱 -> 奖励
    """
    situational_reward = np.zeros(Config.VALUE_NUM, dtype=np.float32)

    if not cur_info or not next_info:
        return situational_reward

    # ---------- 情景1: 危险时冒险收集宝箱 (惩罚) ----------
    cur_danger = cur_info.get('danger_level', 0)
    next_danger = next_info.get('danger_level', 0)
    cur_treasure_dist = cur_info.get('nearest_treasure_dist_norm', 1.0)
    next_treasure_dist = next_info.get('nearest_treasure_dist_norm', 1.0)

    if cur_danger > Config.DANGER_THRESHOLD_HIGH:
        # 危险时靠近宝箱是冒险行为
        if next_treasure_dist < cur_treasure_dist:
            situational_reward[1] += Config.DANGER_TREASURE_PENALTY

    # ---------- 情景2: 安全时不探索 (轻微惩罚) ----------
    if cur_danger < Config.DANGER_THRESHOLD_LOW:
        # 安全时应该探索/收集宝箱
        if next_treasure_dist >= cur_treasure_dist:
            # 没有靠近宝箱
            situational_reward[1] += Config.SAFE_NO_EXPLORATION_PENALTY

    # ---------- 情景3: 成功逃脱 (大奖励) ----------
    escaped = next_info.get('escaped', False)
    if escaped:
        # 从危险逃脱
        nearest_monster_dist = next_info.get('nearest_monster_dist', 100)
        if nearest_monster_dist < 40:  # 近距离逃脱
            situational_reward[0] += Config.PROXIMITY_ESCAPE_REWARD

    # ---------- 情景4: 浪费闪现 (惩罚) ----------
    flash_wasted = next_info.get('flash_wasted', False)
    if flash_wasted:
        situational_reward[0] += Config.FLASH_WASTE_PENALTY_REWARD

    # ---------- 情景5: 使用闪现成功逃脱 (额外奖励) ----------
    flash_used = next_info.get('flash_used', False)
    was_in_danger = cur_info.get('was_in_danger', False)
    if flash_used and was_in_danger and next_danger < Config.DANGER_THRESHOLD_LOW:
        situational_reward[0] += Config.FLASH_ESCAPE_REWARD

    return situational_reward


def calculate_situational_rewards_scalar(cur_info, next_info):
    """
    计算情景奖励 - 标量版本

    Returns:
        float: 情景奖励总和
    """
    total_reward = 0.0

    if not cur_info or not next_info:
        return total_reward

    # ---------- 情景1: 危险时冒险收集宝箱 (惩罚) ----------
    cur_danger = cur_info.get('danger_level', 0)
    next_danger = next_info.get('danger_level', 0)
    cur_treasure_dist = cur_info.get('nearest_treasure_dist_norm', 1.0)
    next_treasure_dist = next_info.get('nearest_treasure_dist_norm', 1.0)

    if cur_danger > Config.DANGER_THRESHOLD_HIGH:
        if next_treasure_dist < cur_treasure_dist:
            total_reward += Config.DANGER_TREASURE_PENALTY * 0.3  # 权重0.3

    # ---------- 情景2: 安全时不探索 (轻微惩罚) ----------
    if cur_danger < Config.DANGER_THRESHOLD_LOW:
        if next_treasure_dist >= cur_treasure_dist:
            total_reward += Config.SAFE_NO_EXPLORATION_PENALTY * 0.3

    # ---------- 情景3: 成功逃脱 (大奖励) ----------
    escaped = next_info.get('escaped', False)
    if escaped:
        nearest_monster_dist = next_info.get('nearest_monster_dist', 100)
        if nearest_monster_dist < 40:
            total_reward += Config.PROXIMITY_ESCAPE_REWARD

    # ---------- 情景4: 浪费闪现 (惩罚) ----------
    flash_wasted = next_info.get('flash_wasted', False)
    if flash_wasted:
        total_reward += Config.FLASH_WASTE_PENALTY_REWARD

    # ---------- 情景5: 使用闪现成功逃脱 (额外奖励) ----------
    flash_used = next_info.get('flash_used', False)
    was_in_danger = cur_info.get('was_in_danger', False)
    if flash_used and was_in_danger and next_danger < Config.DANGER_THRESHOLD_LOW:
        total_reward += Config.FLASH_ESCAPE_REWARD

    return total_reward


def reward_shaping(
    frame_no,
    score,
    terminated,
    truncated,
    remain_info,
    _remain_info,
    obs,
    _obs
):
    """
    奖励塑形函数 - 完整密集奖励版本 (完全复制 agent_target_dqn)

    10个奖励组件：
    1. 终点奖励 (REW_FINISH)
    2. 截断惩罚 (REW_TRUNCATED_PUNISH)
    3. 宝箱奖励/惩罚 (REW_TREASURE)
    4. 闪现距离惩罚 (REW_FLASH)
    5. 撞墙惩罚 (REW_HIT_WALL_PUNISH)
    6. Buff奖励 (REW_BUFF)
    7. 每步惩罚 (REW_EACH_STEP_PUNISH)
    8. 距离奖励 (REW_DISTANCE)
    9. 周围重复步数惩罚 (REW_MEMORY_PUNISH)
    10. 探索奖励 (REW_EXPLORATION)

    Args:
        frame_no: 当前帧号
        score: 环境得分
        terminated: 是否被怪物捕获
        truncated: 是否达到最大步数
        remain_info: 当前帧的预处理信息
        _remain_info: 下一帧的预处理信息
        obs: 当前观测
        _obs: 下一观测

    Returns:
        reward: numpy数组 [总奖励]
    """
    r = 0.0

    if remain_info is None:
        return np.array([0.0], dtype=np.float32)

    # ---------- 1. 终点奖励 ----------
    if terminated:
        r += Config.REW_FINISH

    # ---------- 2. 截断惩罚 ----------
    if truncated:
        r -= Config.REW_TRUNCATED_PUNISH

    # ---------- 3. 宝箱奖励/惩罚 (遗漏惩罚在终局时处理) ----------
    treasure_collected = remain_info.get('treasure_collected', 0)
    r += treasure_collected * Config.REW_TREASURE * 0.1  # 每收集一个宝箱有小奖励

    # ---------- 4. 闪现使用奖励/惩罚 ----------
    flash_used = remain_info.get('flash_used', False)
    danger_level = remain_info.get('danger_level', 0)
    if flash_used:
        if danger_level > Config.DANGER_THRESHOLD_HIGH:
            r += 0.5  # 危险时使用闪现是正确决策
        else:
            r -= 0.1  # 安全时使用闪现略有惩罚

    # ---------- 5. 撞墙惩罚 ----------
    hit_wall = remain_info.get('hit_wall', False)
    if hit_wall:
        r -= Config.REW_HIT_WALL_PUNISH

    # ---------- 6. Buff奖励 ----------
    buff_count = remain_info.get('buff_count', 0)
    r += buff_count * Config.REW_BUFF * 0.1

    # ---------- 7. 每步惩罚 ----------
    r -= Config.REW_EACH_STEP_PUNISH

    # ---------- 8. 距离奖励 (向目标移动) ----------
    # 使用最近的宝箱距离变化
    nearest_treasure_dist = remain_info.get('nearest_treasure_dist', 999)
    if nearest_treasure_dist < 180:
        # 归一化距离奖励 (越近越好)
        r += (1.0 - min(nearest_treasure_dist / 180.0, 1.0)) * 0.01

    # ---------- 9. 周围重复步数惩罚 ----------
    around_memory = remain_info.get('around_memory', np.zeros((Config.REW_MEMORY_PUNISH_SIZE, Config.REW_MEMORY_PUNISH_SIZE)))
    memory_sum = np.sum(around_memory)
    r -= min(
        max(memory_sum - Config.REW_MEMORY_PUNISH_THRESHOLD, 0.0) * Config.REW_MEMORY_PUNISH_COEF,
        1.0
    )

    # ---------- 10. 探索奖励 ----------
    new_explore_grid = remain_info.get('new_explore_grid', 0)
    r += new_explore_grid * Config.REW_EXPLORATION

    # 全局缩放
    r *= Config.REW_GLOBAL_SCALE

    return np.array([r], dtype=np.float32)


# ==================== 样本处理 ====================

def sample_process(list_sample_data, train_step=0):
    """
    样本后处理 - 填充next_value并计算GAE优势函数
    支持动态gamma和lambda

    Args:
        list_sample_data: 单局游戏样本列表
        train_step: 当前训练步数 (用于动态参数)

    Returns:
        处理后的样本列表
    """
    if not list_sample_data:
        return list_sample_data

    # 填充 next_value
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    # 计算 GAE (使用动态参数)
    _calc_gae(list_sample_data, train_step)

    return list_sample_data


def _calc_gae(list_sample_data, train_step=0):
    """
    计算广义优势估计 (Generalized Advantage Estimation)
    支持动态gamma和lambda
    """
    gamma = get_current_gamma(train_step)
    lamda = get_current_lambda(train_step)

    gae = 0.0
    for sample in reversed(list_sample_data):
        # TD误差: r + gamma * V(s') - V(s)
        delta = sample.reward[0] + gamma * sample.next_value[0] - sample.value[0]

        # GAE累加
        gae = delta + gamma * lamda * gae

        # 存储优势函数和回报
        sample.advantage[0] = gae
        sample.reward_sum[0] = gae + sample.value[0]


# ==================== 序列化转换 (用于分布式训练) ====================
# 注意：峡谷追猎是单机训练，这些函数不会被调用
# 移除 @attached 装饰器以避免远程环境导入问题

def SampleData2NumpyData(g_data):
    """将SampleData转换为numpy数组用于网络传输 (与agent_target_dqn一致)"""
    return np.hstack(
        (
            np.array(g_data.obs, dtype=np.float32),
            np.array(g_data._obs, dtype=np.float32),
            np.array(g_data.obs_legal, dtype=np.float32),
            np.array(g_data._obs_legal, dtype=np.float32),
            np.array(g_data.act, dtype=np.float32),
            np.array(g_data.rew, dtype=np.float32),
            np.array(g_data.ret, dtype=np.float32),
            np.array(g_data.done, dtype=np.float32),
        )
    )


def NumpyData2SampleData(s_data):
    """将numpy数组转换回SampleData (与agent_target_dqn一致)"""
    obs_data_size = Config.DIM_OF_OBSERVATION
    legal_data_size = Config.ACTION_NUM
    return SampleData(
        obs=s_data[:obs_data_size],
        _obs=s_data[obs_data_size:2*obs_data_size],
        obs_legal=s_data[2*obs_data_size:2*obs_data_size+legal_data_size],
        _obs_legal=s_data[2*obs_data_size+legal_data_size:2*obs_data_size+2*legal_data_size],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
