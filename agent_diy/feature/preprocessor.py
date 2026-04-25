#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

峡谷追猎 - DIY Agent 特征预处理器
优化版本：课程学习 + 情景判断 + 完整特征工程
"""

import numpy as np
from agent_diy.conf.conf import Config

# 地图常量
MAP_SIZE = Config.MAP_SIZE
MAX_DIST_BUCKET = Config.MAX_DIST_BUCKET
MAX_FLASH_CD = Config.MAX_FLASH_CD
MAX_BUFF_DURATION = Config.MAX_BUFF_DURATION
MAX_MONSTER_SPEED = Config.MAX_MONSTER_SPEED


def _norm(v, v_max, v_min=0.0):
    """将值归一化到 [0, 1]"""
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _calc_distance_bucket(raw_dist):
    """计算距离桶编号 0-5"""
    if raw_dist < 30:
        return 0
    elif raw_dist < 60:
        return 1
    elif raw_dist < 90:
        return 2
    elif raw_dist < 120:
        return 3
    elif raw_dist < 150:
        return 4
    else:
        return 5


def _calc_relative_direction(hero_pos, target_pos):
    """
    计算目标相对于英雄的方位 0-8
    0=重叠, 1=东, 2=东北, 3=北, 4=西北, 5=西, 6=西南, 7=南, 8=东南
    """
    dx = target_pos['x'] - hero_pos['x']
    dz = target_pos['z'] - hero_pos['z']

    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0

    # 计算角度 (弧度)
    angle = np.arctan2(-dz, dx)  # 注意: z轴向下为正，所以取反

    # 转换为 0-8 方向 (每45度一个扇区)
    direction = int((angle + np.pi / 8) / (np.pi / 4)) % 8 + 1
    return direction


class Preprocessor:
    """特征预处理器 - 课程学习 + 情景判断"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置状态"""
        self.step_no = 0
        self.max_step = 1000
        self.last_min_monster_dist_norm = 0.5
        self.last_nearest_treasure_dist_norm = 1.0
        self.treasure_collected = 0
        self.flash_used_in_step = False

        # 历史记录 (用于GRU和时序判断)
        self.position_history = []  # 位置历史
        self.danger_history = []    # 危险等级历史
        self.max_history_len = 10

        # 访问记忆地图 (128x128全局，参考亚军)
        self.visit_memory = np.zeros((128, 128), dtype=np.float32)
        self.hero_pos = (0, 0)  # 当前英雄位置

        # 全局地图状态 (128x128)
        self.global_obstacles = np.full((128, 128), -1.0, np.float32)  # -1=未知
        self.global_treasures = np.zeros((128, 128), np.float32)  # 0.5=大致, 1=精确
        self.global_buffs = np.zeros((128, 128), np.float32)  # -0.5=大致, -1=精确 (负数)

        # 情景判断状态
        self.was_in_danger = False  # 上一步是否在危险中
        self.escape_count = 0       # 成功逃脱次数
        self.flash_waste_count = 0  # 浪费闪现次数

        # 撞墙检测
        self.last_hero_pos = None   # 上一帧英雄位置
        self.last_action = -1       # 上一帧动作

    def feature_process(self, env_obs, last_action):
        """
        处理环境观测，返回特征向量、合法动作掩码、奖励相关信息
        包含课程学习阶段的判断
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)

        # 计算课程学习阶段
        progress_ratio = self.step_no / self.max_step
        is_early_phase = progress_ratio < Config.EARLY_PHASE_RATIO
        is_late_phase = progress_ratio > (1 - Config.LATE_PHASE_RATIO)

        # 获取英雄状态
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]

        # 更新历史
        self.position_history.append((hero_pos['x'], hero_pos['z']))
        if len(self.position_history) > self.max_history_len:
            self.position_history.pop(0)

        # ==================== 英雄特征 (6D) ====================
        hero_feat = self._process_hero_features(hero)

        # ==================== 怪物特征 (6D x 2) ====================
        monsters = frame_state.get("monsters", [])
        monster_feats = self._process_monster_features(monsters, hero_pos)

        # ==================== 宝箱特征 (6D x 10) ====================
        organs = frame_state.get("organs", [])
        treasure_feats, treasure_info = self._process_treasure_features(organs, hero_pos)

        # ==================== Buff特征 (6D x 2) ====================
        buff_feats, buff_info = self._process_buff_features(organs, hero_pos)

        # ==================== 地图特征 (25D) ====================
        map_feat = self._process_map_features(map_info, hero_pos)

        # ==================== 合法动作掩码 (16D) ====================
        legal_action = self._process_legal_actions(legal_act_raw)

        # ==================== 进度特征 (4D) ====================
        progress_feat, danger_level = self._process_progress_features(
            env_info, monsters, hero_pos, progress_ratio
        )

        # 更新危险历史
        self.danger_history.append(danger_level)
        if len(self.danger_history) > self.max_history_len:
            self.danger_history.pop(0)

        # 判断是否成功逃脱
        escaped = self.was_in_danger and danger_level < Config.DANGER_THRESHOLD_LOW
        if escaped:
            self.escape_count += 1

        # 判断闪现是否浪费
        flash_wasted = self.flash_used_in_step and self.was_in_danger == False and danger_level < Config.DANGER_THRESHOLD_LOW
        if flash_wasted:
            self.flash_waste_count += 1

        self.was_in_danger = danger_level > Config.DANGER_THRESHOLD_HIGH

        # 拼接所有特征 (非地图特征)
        feature = np.concatenate([
            hero_feat,
            *monster_feats,
            *treasure_feats,
            *buff_feats,
            map_feat,
            progress_feat,
            np.array(legal_action, dtype=np.float32),
        ])

        # ==================== 构建4通道51x51地图张量 ====================
        map_tensor = self.build_map_tensor(map_info, hero_pos, organs, monsters)

        # 记录是否使用了闪现
        self.flash_used_in_step = (last_action >= 8) if last_action >= 0 else False

        # 撞墙检测：执行了移动动作但位置没变
        hit_wall = False
        if self.last_hero_pos is not None and last_action >= 0 and last_action < 8:
            # 上一步是移动动作(0-7)
            last_pos = self.last_hero_pos
            curr_pos = (hero_pos['x'], hero_pos['z'])
            dist_moved = ((curr_pos[0] - last_pos[0])**2 + (curr_pos[1] - last_pos[1])**2)**0.5
            if dist_moved < 0.5:  # 几乎没移动，认为是撞墙
                hit_wall = True

        # 更新上一帧信息
        self.last_hero_pos = (hero_pos['x'], hero_pos['z'])
        self.last_action = last_action

        # 查找终点信息
        end_info = None
        nearest_end_dist = 999
        for organ in organs:
            if organ.get("sub_type") == 4:  # 终点类型
                end_pos = organ.get("pos", {})
                dist = ((end_pos.get("x", 0) - hero_pos['x'])**2 + (end_pos.get("z", 0) - hero_pos['z'])**2)**0.5
                if dist < nearest_end_dist:
                    nearest_end_dist = dist
                    end_info = organ

        # 构建奖励相关信息 (包含课程学习阶段)
        # 计算周围记忆和
        around_memory_sum = self._calc_around_memory_sum()

        # 计算最近Buff距离 (用于奖励引导)
        nearest_buff_dist = 1.0
        buff_in_view = False
        for b in buff_feats:
            if b[0] > 0:  # is_active > 0
                nearest_buff_dist = b[4]  # dist_norm
                buff_in_view = True
                break

        remain_info = {
            'min_monster_dist_norm': min(
                [m[4] for m in monster_feats if m[0] > 0] or [1.0]
            ),
            'nearest_treasure_dist_norm': treasure_info['nearest_dist_norm'],
            'treasure_collected': treasure_info['collected_count'],
            'danger_level': danger_level,
            'flash_used': self.flash_used_in_step,
            # Buff信息
            'buff_collected': buff_info.get('collected_count', 0),
            'nearest_buff_dist_norm': nearest_buff_dist,
            'nearest_buff_in_view': buff_in_view,
            # 记忆惩罚相关
            'around_memory_sum': around_memory_sum,
            # 课程学习阶段
            'is_early_phase': is_early_phase,
            'is_late_phase': is_late_phase,
            'progress_ratio': progress_ratio,
            # 情景判断
            'escaped': escaped,
            'flash_wasted': flash_wasted,
            'was_in_danger': self.was_in_danger,
            'escape_count': self.escape_count,
            # 撞墙信息
            'hit_wall': hit_wall,
            # 终点信息
            'nearest_end_dist': nearest_end_dist,
            'end_in_view': end_info is not None,
            # 历史信息
            'position_history': self.position_history.copy(),
            'danger_history': self.danger_history.copy(),
            # 怪物具体信息用于情景奖励
            'nearest_monster_dist': min(
                [(m[4], i) for i, m in enumerate(monster_feats) if m[0] > 0],
                key=lambda x: x[0], default=(1.0, -1)
            )[0],
        }

        # 更新上一帧距离
        self.last_min_monster_dist_norm = remain_info['min_monster_dist_norm']
        self.last_nearest_treasure_dist_norm = remain_info['nearest_treasure_dist_norm']

        return feature, legal_action, remain_info, map_tensor

    def _process_hero_features(self, hero):
        """处理英雄特征 (6D)"""
        hero_pos = hero["pos"]

        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)

        flash_cd = hero.get("flash_cooldown", 0)
        flash_cd_norm = _norm(flash_cd, MAX_FLASH_CD)
        flash_ready = 1.0 if flash_cd <= 0 else 0.0

        # 闪现即将可用 (用于决策)
        flash_soon = 1.0 if 0 < flash_cd <= Config.FLASH_CD_THRESHOLD else 0.0

        buff_remain = hero.get("buff_remaining_time", 0)
        buff_remain_norm = _norm(buff_remain, MAX_BUFF_DURATION)
        speed_boost = 1.0 if buff_remain > 0 else 0.0

        return np.array([
            hero_x_norm,
            hero_z_norm,
            flash_cd_norm,
            flash_ready,
            buff_remain_norm,
            speed_boost,
        ], dtype=np.float32)

    def _process_monster_features(self, monsters, hero_pos):
        """处理怪物特征 (6D x 2)"""
        monster_feats = []

        for i in range(Config.MONSTER_COUNT):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]

                if is_in_view > 0:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                    raw_dist = np.sqrt(
                        (hero_pos["x"] - m_pos["x"]) ** 2 +
                        (hero_pos["z"] - m_pos["z"]) ** 2
                    )
                    dist_bucket = _calc_distance_bucket(raw_dist)
                    dist_norm = _norm(dist_bucket, MAX_DIST_BUCKET)

                    rel_dir = _calc_relative_direction(hero_pos, m_pos)
                    rel_dir_norm = _norm(rel_dir, 8)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                    rel_dir_norm = 0.0

                monster_feats.append(np.array([
                    is_in_view,
                    m_x_norm,
                    m_z_norm,
                    m_speed_norm,
                    dist_norm,
                    rel_dir_norm,
                ], dtype=np.float32))
            else:
                monster_feats.append(np.zeros(Config.MONSTER_FEATURE_DIM, dtype=np.float32))

        return monster_feats

    def _process_treasure_features(self, organs, hero_pos):
        """处理宝箱特征 (6D x 10)"""
        treasure_feats = []
        active_treasures = []

        for organ in organs:
            if organ.get("sub_type") == 1:  # 1=宝箱
                active_treasures.append(organ)

        treasure_distances = []
        for t in active_treasures:
            t_pos = t["pos"]
            dist = np.sqrt(
                (hero_pos["x"] - t_pos["x"]) ** 2 +
                (hero_pos["z"] - t_pos["z"]) ** 2
            )
            treasure_distances.append((dist, t))

        treasure_distances.sort(key=lambda x: x[0])
        nearest_treasures = treasure_distances[:Config.TREASURE_COUNT]

        nearest_dist_norm = 1.0
        for i in range(Config.TREASURE_COUNT):
            if i < len(nearest_treasures):
                dist, t = nearest_treasures[i]
                t_pos = t["pos"]

                is_active = 1.0 if t.get("status") == 1 else 0.0
                t_x_norm = _norm(t_pos["x"], MAP_SIZE)
                t_z_norm = _norm(t_pos["z"], MAP_SIZE)

                dist_bucket = _calc_distance_bucket(dist)
                dist_norm = _norm(dist_bucket, MAX_DIST_BUCKET)

                if i == 0:
                    nearest_dist_norm = dist_norm

                rel_dir = _calc_relative_direction(hero_pos, t_pos)
                rel_dir_norm = _norm(rel_dir, 8)

                priority = 1.0 - dist_norm

                treasure_feats.append(np.array([
                    is_active,
                    t_x_norm,
                    t_z_norm,
                    dist_norm,
                    rel_dir_norm,
                    priority,
                ], dtype=np.float32))
            else:
                treasure_feats.append(np.zeros(Config.TREASURE_FEATURE_DIM, dtype=np.float32))

        info = {
            'nearest_dist_norm': nearest_dist_norm,
            'collected_count': Config.TREASURE_COUNT - len(active_treasures),
        }

        return treasure_feats, info

    def _process_buff_features(self, organs, hero_pos):
        """处理Buff特征 (6D x 2)"""
        buff_feats = []
        active_buffs = []

        for organ in organs:
            if organ.get("sub_type") == 2:  # 2=加速buff
                active_buffs.append(organ)

        buff_distances = []
        for b in active_buffs:
            b_pos = b["pos"]
            dist = np.sqrt(
                (hero_pos["x"] - b_pos["x"]) ** 2 +
                (hero_pos["z"] - b_pos["z"]) ** 2
            )
            buff_distances.append((dist, b))

        buff_distances.sort(key=lambda x: x[0])
        nearest_buffs = buff_distances[:Config.BUFF_COUNT]

        for i in range(Config.BUFF_COUNT):
            if i < len(nearest_buffs):
                dist, b = nearest_buffs[i]
                b_pos = b["pos"]

                is_active = 1.0 if b.get("status") == 1 else 0.0
                b_x_norm = _norm(b_pos["x"], MAP_SIZE)
                b_z_norm = _norm(b_pos["z"], MAP_SIZE)

                dist_bucket = _calc_distance_bucket(dist)
                dist_norm = _norm(dist_bucket, MAX_DIST_BUCKET)

                rel_dir = _calc_relative_direction(hero_pos, b_pos)
                rel_dir_norm = _norm(rel_dir, 8)
                remaining_time_norm = 1.0

                buff_feats.append(np.array([
                    is_active,
                    b_x_norm,
                    b_z_norm,
                    dist_norm,
                    rel_dir_norm,
                    remaining_time_norm,
                ], dtype=np.float32))
            else:
                buff_feats.append(np.zeros(Config.BUFF_FEATURE_DIM, dtype=np.float32))

        # 统计已收集Buff数量 (总数2 - 剩余数)
        collected_count = Config.BUFF_COUNT - len(active_buffs)

        return buff_feats, {'collected_count': collected_count}

    def _calc_around_memory_sum(self):
        """计算周围3x3区域的访问次数总和"""
        hx, hz = self.hero_pos
        offset = Config.REW_MEMORY_PUNISH_SIZE // 2  # 1 for size 3

        total = 0.0
        for dz in range(-offset, offset + 1):
            for dx in range(-offset, offset + 1):
                x = hx + dx
                z = hz + dz
                if 0 <= x < 128 and 0 <= z < 128:
                    total += self.visit_memory[z, x]

        return total

    def _process_map_features(self, map_info, hero_pos):
        """处理局部地图特征 (25D) - 5x5区域"""
        map_feat = np.zeros(Config.MAP_LOCAL_DIM, dtype=np.float32)

        if map_info is None or len(map_info) == 0:
            return map_feat

        center = len(map_info) // 2
        offset = 2

        flat_idx = 0
        for row in range(center - offset, center + offset + 1):
            for col in range(center - offset, center + offset + 1):
                if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                    map_feat[flat_idx] = float(map_info[row][col] != 0)
                flat_idx += 1

        return map_feat

    def _update_global_maps(self, hero_pos, map_info, organs):
        """更新128x128全局地图 (参考亚军)"""
        hx, hz = int(hero_pos['x']), int(hero_pos['z'])
        self.hero_pos = (hx, hz)

        # 更新访问记忆
        if 0 <= hx < 128 and 0 <= hz < 128:
            self.visit_memory[hz, hx] += 1.0

        # 更新障碍物 (21x21视野)
        if map_info is not None:
            local_center = len(map_info) // 2
            for i in range(len(map_info)):
                for j in range(len(map_info[0])):
                    gx = hx + i - local_center
                    gz = hz + j - local_center
                    if 0 <= gx < 128 and 0 <= gz < 128:
                        # 1=可通行, 0=障碍
                        self.global_obstacles[gz, gx] = float(map_info[i][j] != 0)

        # 更新宝箱和Buff位置
        for organ in organs:
            sub_type = organ.get("sub_type")
            pos = organ.get("pos", {})
            ox, oz = int(pos.get("x", 0)), int(pos.get("z", 0))
            is_in_view = organ.get("is_in_view", 0)
            status = organ.get("status", 0)

            # 视野外时，用relative_pos估算
            if is_in_view == 0 and 'relative_pos' in organ:
                rel = organ['relative_pos']
                dist_bucket = rel.get('l2_distance', 5)
                direction = rel.get('direction', 0)
                # 估算距离和位置
                est_dist = [15, 45, 75, 105, 135, 165][min(dist_bucket, 5)]
                angle = (direction - 1) * 45 if direction > 0 else 0
                import math
                dx = est_dist * math.cos(math.radians(angle))
                dz = est_dist * math.sin(math.radians(angle))
                ox = int(hx + dx)
                oz = int(hz + dz)

            if 0 <= ox < 128 and 0 <= oz < 128:
                if sub_type == 1:  # 宝箱
                    if is_in_view > 0 and status == 1:
                        self.global_treasures[oz, ox] = 1.0  # 精确
                    else:
                        self.global_treasures[oz, ox] = max(self.global_treasures[oz, ox], 0.5)  # 大致
                elif sub_type == 2:  # Buff (负数)
                    if is_in_view > 0 and status == 1:
                        self.global_buffs[oz, ox] = -1.0  # 精确
                    else:
                        self.global_buffs[oz, ox] = min(self.global_buffs[oz, ox], -0.5)  # 大致

    def build_map_tensor(self, map_info, hero_pos, organs, monsters):
        """
        构建4通道51x51地图张量 (从128x128全局地图裁剪，参考亚军)

        通道0: 障碍物 (-1=未知, 0=障碍, 1=通路)
        通道1: 访问记忆 (0-1, 访问次数归一化)
        通道2: 宝箱+Buff (0=无, 0.5=宝箱大致, 1=宝箱精确, -0.5=Buff大致, -1=Buff精确)
        通道3: 怪物 (0=无, 0.5=大致, 1=精确)

        Args:
            map_info: 局部地图信息 (21x21)
            hero_pos: 英雄位置 {'x', 'z'}
            organs: 物件列表 (宝箱、Buff)
            monsters: 怪物列表

        Returns:
            map_tensor: [4, 51, 51] numpy数组
        """
        # 首先更新全局地图
        self._update_global_maps(hero_pos, map_info, organs)

        # 从128x128全局地图裁剪51x51区域
        hx, hz = int(hero_pos['x']), int(hero_pos['z'])
        half_size = 25  # 51//2

        map_tensor = np.zeros((4, 51, 51), dtype=np.float32)

        for i in range(51):
            for j in range(51):
                # 全局坐标
                gx = hx + j - half_size
                gz = hz + i - half_size

                if 0 <= gx < 128 and 0 <= gz < 128:
                    # 通道0: 障碍物
                    map_tensor[0, i, j] = self.global_obstacles[gz, gx]

                    # 通道1: 访问记忆 (归一化)
                    map_tensor[1, i, j] = min(self.visit_memory[gz, gx] / 10.0, 1.0)

                    # 通道2: 宝箱+Buff (合并，符号区分)
                    # 正数=宝箱, 负数=Buff
                    treasure_val = self.global_treasures[gz, gx]
                    buff_val = self.global_buffs[gz, gx]
                    if treasure_val > 0:
                        map_tensor[2, i, j] = treasure_val
                    elif buff_val < 0:
                        map_tensor[2, i, j] = buff_val

                    # 通道3: 怪物
                    # 简化处理：在视野内的怪物标记为1
                    for m in monsters:
                        if m.get('is_in_view', 0):
                            mx = int(m['pos']['x'])
                            mz = int(m['pos']['z'])
                            if gx == mx and gz == mz:
                                map_tensor[3, i, j] = 1.0
                                break
                else:
                    # 超出地图范围标记为-1 (未知)
                    map_tensor[0, i, j] = -1.0

        return map_tensor

    def _process_legal_actions(self, legal_act_raw):
        """处理合法动作掩码 (16D)"""
        legal_action = [1] * Config.ACTION_NUM

        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(Config.ACTION_NUM, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < Config.ACTION_NUM}
                legal_action = [1 if j in valid_set else 0 for j in range(Config.ACTION_NUM)]

        if sum(legal_action) == 0:
            legal_action = [1] * Config.ACTION_NUM

        return legal_action

    def _process_progress_features(self, env_info, monsters, hero_pos, progress_ratio):
        """处理进度特征 (4D) - 包含课程学习阶段"""
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm

        monster_interval = env_info.get("monster_interval", 300)
        monster2_timer = _norm(max(0, monster_interval - self.step_no), monster_interval)

        # 危险等级计算
        danger_level = 0.0
        min_dist = float('inf')
        monster_speed = 1

        if monsters:
            for m in monsters:
                if m.get("is_in_view", 0):
                    m_pos = m["pos"]
                    dist = np.sqrt(
                        (hero_pos["x"] - m_pos["x"]) ** 2 +
                        (hero_pos["z"] - m_pos["z"]) ** 2
                    )
                    if dist < min_dist:
                        min_dist = dist
                        monster_speed = m.get("speed", 1)

        # 危险等级: 综合考虑距离和怪物速度
        if min_dist < 30:
            danger_level = 1.0
        elif min_dist < 50 and monster_speed >= 2:
            danger_level = 0.9  # 高速怪物近距离更危险
        elif min_dist < 60:
            danger_level = 0.7
        elif min_dist < 90:
            danger_level = 0.4
        elif min_dist < 120:
            danger_level = 0.2
        else:
            danger_level = 0.0

        progress_feat = np.array([
            step_norm,
            survival_ratio,
            monster2_timer,
            danger_level,
        ], dtype=np.float32)

        return progress_feat, danger_level
