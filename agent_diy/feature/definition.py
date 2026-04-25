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
    奖励塑形函数 - 课程学习 + 情景奖励 + 密集奖励

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
        reward: numpy数组 [生存奖励, 宝箱奖励]
    """
    reward = np.zeros(Config.VALUE_NUM, dtype=np.float32)

    if not remain_info:
        remain_info = {}
    if not _remain_info:
        _remain_info = {}

    # 获取课程学习阶段
    is_early_phase = remain_info.get('is_early_phase', False)
    is_late_phase = remain_info.get('is_late_phase', False)
    progress_ratio = remain_info.get('progress_ratio', 0.5)

    # ============ 生存价值奖励 ============
    survive_reward = Config.SURVIVE_REWARD_BASE

    # 距离塑形: 远离怪物获得正奖励
    cur_min_dist = remain_info.get('min_monster_dist_norm', 0.5)
    next_min_dist = _remain_info.get('min_monster_dist_norm', 0.5)
    dist_shaping = Config.DISTANCE_SHAPING_COEF * (next_min_dist - cur_min_dist)

    # 步数奖励: 活得越久奖励越高 (鼓励生存)
    step_reward = Config.STEP_REWARD_COEF * progress_ratio

    # 终局奖励
    if terminated:
        final_reward = Config.DEATH_PENALTY
    elif truncated:
        final_reward = Config.WIN_REWARD
    else:
        final_reward = 0.0

    # 基础生存奖励
    survive_total = survive_reward + dist_shaping + step_reward + final_reward

    # 撞墙惩罚
    hit_wall = remain_info.get('hit_wall', False)
    if hit_wall:
        survive_total += Config.REW_HIT_WALL

    # ============ 宝箱收集奖励 (课程学习) ============
    treasure_reward = 0.0

    # 根据阶段调整宝箱奖励权重
    if is_early_phase:
        treasure_weight = 1.0  # 前期重视收集
    elif is_late_phase:
        treasure_weight = 0.3  # 后期降低收集权重，重视生存
    else:
        treasure_weight = 0.7  # 中期平衡

    # 收集宝箱奖励
    cur_treasure_count = remain_info.get('treasure_collected', 0)
    next_treasure_count = _remain_info.get('treasure_collected', 0)
    if next_treasure_count > cur_treasure_count:
        if is_early_phase:
            treasure_reward += Config.TREASURE_REWARD_EARLY
        else:
            treasure_reward += Config.TREASURE_REWARD_LATE

    # 探索奖励: 靠近未收集宝箱
    cur_treasure_dist = remain_info.get('nearest_treasure_dist_norm', 1.0)
    next_treasure_dist = _remain_info.get('nearest_treasure_dist_norm', 1.0)
    if next_treasure_dist < cur_treasure_dist:
        treasure_reward += Config.EXPLORATION_REWARD * (cur_treasure_dist - next_treasure_dist)

    treasure_total = treasure_reward * treasure_weight

    # ============ Buff奖励 (递减，参考优秀经验) ============
    buff_reward = 0.0
    cur_buff_count = remain_info.get('buff_collected', 0)
    next_buff_count = _remain_info.get('buff_collected', 0)
    if next_buff_count > cur_buff_count:
        # 递减奖励: 0.5, 0.25, 0.125...
        decay = Config.REW_BUFF_DECAY ** cur_buff_count
        buff_reward = Config.REW_BUFF * decay

    # ============ 距离奖励 (宝箱 + Buff) ============
    distance_reward = 0.0

    # 宝箱距离奖励
    if 'nearest_treasure_dist_norm' in remain_info and 'nearest_treasure_dist_norm' in _remain_info:
        cur_treasure_dist = remain_info['nearest_treasure_dist_norm']
        next_treasure_dist = _remain_info['nearest_treasure_dist_norm']
        delta_dist = cur_treasure_dist - next_treasure_dist  # 靠近为正
        delta_dist = np.clip(delta_dist, -Config.REW_DISTANCE_CLIP, Config.REW_DISTANCE_CLIP)
        distance_reward += delta_dist * Config.REW_DISTANCE

    # Buff距离奖励 (新增：引导agent去收集Buff)
    # 假设buff特征中第4个元素(索引3)是距离桶归一化值
    buff_in_view = remain_info.get('nearest_buff_in_view', False)
    if buff_in_view and 'nearest_buff_dist_norm' in remain_info and 'nearest_buff_dist_norm' in _remain_info:
        cur_buff_dist = remain_info['nearest_buff_dist_norm']
        next_buff_dist = _remain_info['nearest_buff_dist_norm']
        buff_delta = cur_buff_dist - next_buff_dist
        buff_delta = np.clip(buff_delta, -Config.REW_DISTANCE_CLIP, Config.REW_DISTANCE_CLIP)
        distance_reward += buff_delta * Config.REW_DISTANCE * 1.5  # Buff距离奖励权重更高

    # 终点距离奖励 (引导agent去终点) - 始终生效，确保agent知道要去终点
    if 'nearest_end_dist' in remain_info and 'nearest_end_dist' in _remain_info:
        cur_end_dist = remain_info['nearest_end_dist']
        next_end_dist = _remain_info['nearest_end_dist']
        # 归一化距离到0-1范围
        cur_end_dist_norm = min(cur_end_dist / 180.0, 1.0)
        next_end_dist_norm = min(next_end_dist / 180.0, 1.0)
        end_delta = cur_end_dist_norm - next_end_dist_norm  # 靠近为正
        # 始终给予较强的终点引导 (确保agent知道要去哪)
        distance_reward += end_delta * Config.REW_END_DISTANCE
        # 额外的接近终点奖励 (越近奖励越高)
        if next_end_dist < 50:  # 接近终点
            distance_reward += 0.1 * (1 - next_end_dist / 50)

    # ============ 记忆惩罚 (避免重复路径) ============
    memory_penalty = 0.0
    if 'around_memory_sum' in remain_info:
        around_sum = remain_info['around_memory_sum']
        if around_sum > Config.REW_MEMORY_PUNISH_THRESHOLD:
            memory_penalty = min(
                (around_sum - Config.REW_MEMORY_PUNISH_THRESHOLD) * Config.REW_MEMORY_PUNISH_COEF,
                1.0
            )

    # ============ 情景奖励 (核心) ============
    situational_total = calculate_situational_rewards_scalar(remain_info, _remain_info)

    # 总奖励 = 生存 + 宝箱 + Buff + 距离 - 记忆惩罚 + 情景
    # 注意：前期重视收集，后期重视生存
    if is_early_phase:
        collection_weight = 1.0
        survival_weight = 0.5
    elif is_late_phase:
        collection_weight = 0.3
        survival_weight = 1.5
    else:
        collection_weight = 0.7
        survival_weight = 1.0

    total_reward = (
        survive_total * survival_weight +
        treasure_total * collection_weight +
        buff_reward * collection_weight +
        distance_reward -
        memory_penalty +
        situational_total
    )
    return np.array([total_reward], dtype=np.float32)


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
