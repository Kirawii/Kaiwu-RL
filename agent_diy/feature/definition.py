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
    action=None,         # 采样动作
    d_action=None,       # 贪心动作 (argmax)
    prob=None,           # 动作概率分布
    value=None,          # 状态价值 (多头)
)


SampleData = create_cls(
    "SampleData",
    obs=Config.FEATURE_DIM,           # 观测特征维度
    legal_action=Config.ACTION_NUM,    # 合法动作掩码维度
    act=1,                             # 执行动作
    prob=Config.ACTION_NUM,            # 动作概率
    reward=1,                          # 即时奖励 (标量)
    value=1,                           # 状态价值 (标量)
    next_value=1,                      # 下一状态价值
    advantage=1,                       # 优势函数
    reward_sum=1,                      # 回报和
    done=1,                            # 是否结束
    map_tensor=None,                   # 4通道51x51地图张量 (可选)
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
    奖励塑形函数 - 稀疏奖励版本 (参考亚军方案 kaiwu_taichu-2025)

    简化策略：
    1. 每步小额惩罚 (-0.02)，鼓励高效行动
    2. 只有终局有大奖励/惩罚
    3. 让模型自己学习生存和收集策略

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
    # ============ 稀疏奖励核心：只有基础惩罚 + 终局奖励 ============

    # 每步惩罚（很小，让模型有紧迫感但不过度惩罚生存）
    # 亚军方案: -0.02/步
    r = -0.02

    # 撞墙惩罚（可选，如果信息可用）
    # hit_wall = remain_info.get('hit_wall', False)
    # if hit_wall:
    #     r -= 0.1

    # 终局奖励（稀疏奖励的主要来源）
    if terminated:
        r += Config.DEATH_PENALTY  # -10
    elif truncated:
        r += Config.WIN_REWARD     # +10

    # ============ 注释掉的密集奖励（供参考） ============
    """
    # 以下奖励被注释掉，采用稀疏奖励策略：
    # - 宝箱收集奖励
    # - Buff收集奖励
    # - 距离塑形奖励
    # - 情景奖励
    # - 记忆惩罚
    #
    # 原理：让模型自己从终局奖励中学习长期策略，
    # 而不是被人为设计的密集奖励引导
    """

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

def SampleData2NumpyData(sample):
    """将SampleData转换为numpy数组用于网络传输"""
    data = np.concatenate([
        sample.obs.flatten(),
        sample.legal_action.flatten(),
        np.array([sample.act]),
        sample.prob.flatten(),
        sample.reward.flatten(),
        sample.value.flatten(),
        sample.next_value.flatten(),
        sample.advantage.flatten(),
        sample.reward_sum.flatten(),
        np.array([sample.done]),
    ])
    return data


def NumpyData2SampleData(data):
    """将numpy数组转换回SampleData"""
    idx = 0

    obs = data[idx:idx + Config.FEATURE_DIM].astype(np.float32)
    idx += Config.FEATURE_DIM

    legal_action = data[idx:idx + Config.ACTION_NUM].astype(np.float32)
    idx += Config.ACTION_NUM

    act = int(data[idx])
    idx += 1

    prob = data[idx:idx + Config.ACTION_NUM].astype(np.float32)
    idx += Config.ACTION_NUM

    reward = data[idx:idx + 1].astype(np.float32)
    idx += 1

    value = data[idx:idx + 1].astype(np.float32)
    idx += 1

    next_value = data[idx:idx + 1].astype(np.float32)
    idx += 1

    advantage = data[idx:idx + 1].astype(np.float32)
    idx += 1

    reward_sum = data[idx:idx + 1].astype(np.float32)
    idx += 1

    done = float(data[idx])

    return SampleData(
        obs=obs,
        legal_action=legal_action,
        act=act,
        prob=prob,
        reward=reward,
        value=value,
        next_value=next_value,
        advantage=advantage,
        reward_sum=reward_sum,
        done=done,
    )
