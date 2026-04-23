#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

峡谷追猎 - DIY Agent 特征预处理器
完全参考 agent_target_dqn 亚军方案

特征格式:
- 英雄特征 (5D): pos_x_norm, pos_z_norm, flash_status, flash_cd_norm, buff_remain_time_norm
- 地图特征 (4x51x51 = 10404D): 障碍物、访问记忆、宝箱/Buff、终点
"""

import numpy as np
import math
from agent_diy.conf.conf import Config


class Preprocessor:
    """
    特征预处理器 - 完全复制agent_target_dqn的StateManager逻辑
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置状态"""
        self.step = 0
        self.max_step = 1000
        self.last_action = -1

        # 128x128全局地图 (参考agent_target_dqn)
        self.obstacles = np.full((128, 128), -1.0, np.float32)  # -1=未知, 0=障碍, 1=通路
        self.memory = np.zeros((128, 128), np.float32)  # 访问次数
        self.buff_treasures = np.zeros((128, 128), np.float32)  # 宝箱/Buff
        self.end = np.zeros((128, 128), np.float32)  # 终点

        # 物件位置记录 (用于更新地图)
        self.avail_treasures = [False] * 13
        self.avail_buff = False
        self.last_pos_treasures = [None] * 13
        self.last_pos_buff = None
        self.last_pos_end = None

        # 英雄位置
        self.hero_pos = np.array([0, 0], np.int32)

        # 统计
        self.buff_count = 0
        self.treasure_collected = 0
        self.hit_wall = False
        self.last_hero_pos = None

        # 闪现相关
        self.talent_max_cd = 0

    def feature_process(self, env_obs, last_action):
        """
        处理环境观测，返回特征向量和合法动作掩码

        Args:
            env_obs: 环境观测
            last_action: 上一帧动作

        Returns:
            feature: [DIM_OF_OBSERVATION] 特征向量 (5 + 10404 = 10409D)
            legal_action: [ACTION_NUM] 合法动作掩码
            remain_info: 额外信息用于奖励计算
            map_tensor: [4, 51, 51] 地图张量 (与特征中的地图部分一致)
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)
        self.last_action = last_action

        # 获取英雄信息 (按照官方文档: heroes是HeroState对象)
        # 官方文档HeroState字段: hero_id, pos, treasure_score, step_score, treasure_collected_count
        hero_info = frame_state.get("heroes", {})

        # 确保必要字段存在
        if not hero_info or "pos" not in hero_info:
            hero_info = {"pos": {"x": 64, "z": 64}}

        hero_pos = np.array([hero_info["pos"]["x"], hero_info["pos"]["z"]], np.float32)

        # 官方文档 HeroState 只有: hero_id, pos, treasure_score, step_score, treasure_collected_count
        # 没有 talent 和 buff_remain_time 字段
        # 从 env_info 获取闪现信息 (官方文档规定)
        flash_cooldown = env_info.get("flash_cooldown", 0)
        collected_buff = env_info.get("collected_buff", 0)

        # 更新闪现最大cd (用于归一化)
        self.talent_max_cd = max(self.talent_max_cd, flash_cooldown)

        # 更新地图
        self._update_map(hero_pos, map_info, frame_state.get("organs", []))

        # 计算英雄特征 (5D) - 传入env_info获取flash信息
        hero_feature = self._get_hero_feature(hero_info, flash_cooldown, collected_buff)

        # 计算周围51x51地图特征 (4x51x51 = 10404D)
        map_feature = self._get_around_feature(size=51)

        # 拼接特征: [5D英雄 + 10404D地图]
        feature = np.concatenate([hero_feature, map_feature.flatten()], dtype=np.float32)

        # 计算合法动作掩码
        legal_action = self._get_legal_actions(legal_act_raw, flash_cooldown)

        # 计算撞墙
        self.hit_wall = self._check_hit_wall(hero_pos)
        self.last_hero_pos = hero_pos.copy()

        # 构建remain_info用于奖励计算
        remain_info = self._build_remain_info(env_info, hero_info, hero_pos)

        # 地图张量 (4, 51, 51)
        map_tensor = self._get_map_tensor(size=51)

        return feature, legal_action, remain_info, map_tensor

    def _get_hero_feature(self, hero_info, flash_cooldown=0, collected_buff=0):
        """
        计算英雄特征 (5D)
        1. pos_x_norm: x坐标归一化到(-1, 1)
        2. pos_z_norm: z坐标归一化到(-1, 1)
        3. flash_status: 闪现是否可用 (0或1) - cooldown==0时可用
        4. flash_cd_norm: 闪现cd归一化到(0, 1)
        5. buff_count_norm: buff收集次数归一化
        """
        pos = hero_info["pos"]

        # 归一化到(-1, 1)
        pos_x_norm = (pos["x"] - 64) / 64.0  # 128x128地图中心在64
        pos_z_norm = (pos["z"] - 64) / 64.0

        # 从 env_info 获取的 flash_cooldown
        flash_status = 1.0 if flash_cooldown == 0 else 0.0
        flash_cd_norm = flash_cooldown / self.talent_max_cd if self.talent_max_cd > 0 else 0.0

        # buff收集次数 (官方文档没有buff_remain_time，用collected_buff代替)
        buff_count_norm = min(collected_buff / 2.0, 1.0)  # 假设最多2个buff

        return np.array([
            pos_x_norm,
            pos_z_norm,
            flash_status,
            flash_cd_norm,
            buff_count_norm,
        ], dtype=np.float32)

    def _update_map(self, hero_pos, map_info, organs):
        """更新128x128全局地图"""
        self.hero_pos = hero_pos.astype(np.int32)

        # 更新访问记忆
        hx, hz = self.hero_pos
        if 0 <= hx < 128 and 0 <= hz < 128:
            self.memory[hz, hx] += 1.0

        # 更新障碍物和新探索区域
        self._update_obstacles(hero_pos, map_info)

        # 更新物件 (宝箱、Buff、终点)
        for organ in organs:
            sub_type = organ.get("sub_type")
            if sub_type == 1:  # 宝箱
                self._update_treasure(organ, hero_pos)
            elif sub_type == 2:  # Buff
                self._update_buff(organ, hero_pos)
            elif sub_type == 4:  # 终点
                self._update_end(organ, hero_pos)

    def _update_obstacles(self, hero_pos, map_info):
        """更新障碍物信息"""
        if map_info is None or len(map_info) == 0:
            return

        hero_pos = hero_pos.astype(np.int32)

        # 按照官方文档: map_info是int32[][]二维数组
        # 但某些版本可能是字典列表格式，需要兼容处理
        if isinstance(map_info[0], dict) and "values" in map_info[0]:
            # 字典列表格式 (demo中的格式)
            map_array = np.array([line["values"] for line in map_info], np.float32)
        elif isinstance(map_info, list) and isinstance(map_info[0], list):
            # 直接的二维数组格式 (官方文档格式)
            map_array = np.array(map_info, np.float32)
        else:
            # 未知格式，尝试直接转换
            map_array = np.array(map_info, np.float32)

        map_array = np.clip(map_array, 0, 1)
        map_size = map_array.shape[0]

        for i in range(map_size):
            for j in range(map_size):
                # map_array[i, j]: i是行(z), j是列(x)
                # obstacles[v, u]: v是z, u是x
                u = hero_pos[0] + j - map_size // 2  # x方向用列索引j
                v = hero_pos[1] + i - map_size // 2  # z方向用行索引i
                if 0 <= u < 128 and 0 <= v < 128:
                    self.obstacles[v, u] = map_array[i, j]

    def _update_treasure(self, organ, hero_pos):
        """更新宝箱位置"""
        organ_id = organ["config_id"] - 1  # 0-12
        pos = np.array([organ["pos"]["x"], organ["pos"]["z"]], np.float32)
        pos_round = pos.round().astype(np.int32)

        def clean_last_pos():
            if self.last_pos_treasures[organ_id] is not None:
                lx, lz = self.last_pos_treasures[organ_id]
                if 0 <= lx < 128 and 0 <= lz < 128:
                    self.buff_treasures[lz, lx] = 0.0
            self.last_pos_treasures[organ_id] = pos_round

        # 不可获取 (已收集)
        if organ["status"] == 0:
            if self.avail_treasures[organ_id]:
                if 0 <= pos_round[0] < 128 and 0 <= pos_round[1] < 128:
                    self.buff_treasures[pos_round[1], pos_round[0]] = 0.0
                self.treasure_collected += 1
            self.avail_treasures[organ_id] = False
            return

        self.avail_treasures[organ_id] = True

        # 判断是否已精确定位
        if organ["status"] == -1:  # 视野外，大致位置
            rel_pos = organ.get("relative_pos", {})
            direction = rel_pos.get("direction", 0)
            distance_bucket = rel_pos.get("l2_distance", 3)

            # 估算距离
            est_dist = [10, 30, 60, 90, 120, 150][min(distance_bucket, 5)]
            angle = (direction - 1) * 45 if direction > 0 else 0
            dx = est_dist * math.cos(math.radians(angle))
            dz = est_dist * math.sin(math.radians(angle))
            est_pos = np.array([
                max(0, min(127, hero_pos[0] + dx)),
                max(0, min(127, hero_pos[1] + dz)),
            ], np.float32)
            pos_round = est_pos.round().astype(np.int32)

            clean_last_pos()
            if 0 <= pos_round[0] < 128 and 0 <= pos_round[1] < 128:
                self.buff_treasures[pos_round[1], pos_round[0]] = 0.5
        else:  # 视野内，精确位置
            clean_last_pos()
            if 0 <= pos_round[0] < 128 and 0 <= pos_round[1] < 128:
                self.buff_treasures[pos_round[1], pos_round[0]] = 1.0

    def _update_buff(self, organ, hero_pos):
        """更新Buff位置"""
        pos = np.array([organ["pos"]["x"], organ["pos"]["z"]], np.float32)
        pos_round = pos.round().astype(np.int32)

        def clean_last_pos():
            if self.last_pos_buff is not None:
                lx, lz = self.last_pos_buff
                if 0 <= lx < 128 and 0 <= lz < 128:
                    self.buff_treasures[lz, lx] = 0.0
            self.last_pos_buff = pos_round

        # 不可获取 (已收集)
        if organ["status"] == 0:
            if self.avail_buff:
                if 0 <= pos_round[0] < 128 and 0 <= pos_round[1] < 128:
                    self.buff_treasures[pos_round[1], pos_round[0]] = 0.0
                self.buff_count += 1
            self.avail_buff = False
            return

        self.avail_buff = True

        if organ["status"] == -1:  # 视野外
            rel_pos = organ.get("relative_pos", {})
            direction = rel_pos.get("direction", 0)
            distance_bucket = rel_pos.get("l2_distance", 3)

            est_dist = [10, 30, 60, 90, 120, 150][min(distance_bucket, 5)]
            angle = (direction - 1) * 45 if direction > 0 else 0
            dx = est_dist * math.cos(math.radians(angle))
            dz = est_dist * math.sin(math.radians(angle))
            est_pos = np.array([
                max(0, min(127, hero_pos[0] + dx)),
                max(0, min(127, hero_pos[1] + dz)),
            ], np.float32)
            pos_round = est_pos.round().astype(np.int32)

            clean_last_pos()
            if 0 <= pos_round[0] < 128 and 0 <= pos_round[1] < 128:
                self.buff_treasures[pos_round[1], pos_round[0]] = -0.5
        else:  # 视野内
            clean_last_pos()
            if 0 <= pos_round[0] < 128 and 0 <= pos_round[1] < 128:
                self.buff_treasures[pos_round[1], pos_round[0]] = -1.0

    def _update_end(self, organ, hero_pos):
        """更新终点位置"""
        pos = np.array([organ["pos"]["x"], organ["pos"]["z"]], np.float32)
        pos_round = pos.round().astype(np.int32)

        def clean_last_pos():
            if self.last_pos_end is not None:
                lx, lz = self.last_pos_end
                if 0 <= lx < 128 and 0 <= lz < 128:
                    self.end[lz, lx] = 0.0
            self.last_pos_end = pos_round

        if organ["status"] == -1:  # 视野外
            rel_pos = organ.get("relative_pos", {})
            direction = rel_pos.get("direction", 0)
            distance_bucket = rel_pos.get("l2_distance", 3)

            est_dist = [10, 30, 60, 90, 120, 150][min(distance_bucket, 5)]
            angle = (direction - 1) * 45 if direction > 0 else 0
            dx = est_dist * math.cos(math.radians(angle))
            dz = est_dist * math.sin(math.radians(angle))
            est_pos = np.array([
                max(0, min(127, hero_pos[0] + dx)),
                max(0, min(127, hero_pos[1] + dz)),
            ], np.float32)
            pos_round = est_pos.round().astype(np.int32)

            clean_last_pos()
            if 0 <= pos_round[0] < 128 and 0 <= pos_round[1] < 128:
                self.end[pos_round[1], pos_round[0]] = 0.5
        else:  # 视野内
            clean_last_pos()
            if 0 <= pos_round[0] < 128 and 0 <= pos_round[1] < 128:
                self.end[pos_round[1], pos_round[0]] = 1.0

    def _get_around_feature(self, size=51):
        """
        获取周围size x size的特征 (展平为向量)
        返回: [4, size, size] 展平后的向量 (10404D)
        """
        assert size % 2 == 1, f"size must be odd, got {size}"

        x = np.zeros((4, size, size), np.float32)

        for i in range(size):
            for j in range(size):
                ii = self.hero_pos[0] + i - size // 2
                jj = self.hero_pos[1] + j - size // 2

                if 0 <= ii < 128 and 0 <= jj < 128:
                    x[0, i, j] = self.obstacles[jj, ii]
                    x[1, i, j] = min(self.memory[jj, ii] / 10.0, 1.0)

        # 处理宝箱/Buff和终点 (超出边界的需要映射到边界)
        def cvt_pos_to_bound(pos):
            """将超出范围的treasure,buff,end转为的可用边界"""
            center = np.array([size // 2, size // 2], np.int32)
            delta_pos = pos - self.hero_pos

            if abs(delta_pos[0]) <= size // 2 and abs(delta_pos[1]) <= size // 2:
                return delta_pos + center

            theta = math.atan2(delta_pos[1], delta_pos[0])
            if abs(delta_pos[0]) > abs(delta_pos[1]):
                delta_pos[0] = size // 2 * np.sign(delta_pos[0])
                delta_pos[1] = round(size // 2 * abs(math.tan(theta)) * np.sign(delta_pos[1]))
            else:
                delta_pos[1] = size // 2 * np.sign(delta_pos[1])
                delta_pos[0] = round(size // 2 / abs(math.tan(theta)) * np.sign(delta_pos[0]))
            return center + delta_pos.astype(np.int32)

        # 宝箱 (np.argwhere返回[row, col]即[z, x])
        treasures_pos = np.argwhere(self.buff_treasures > 0)
        for pos in treasures_pos:
            # pos是[z, x]，转换为[x, z]给cvt_pos_to_bound
            pos_xz = np.array([pos[1], pos[0]], np.int32)
            p = cvt_pos_to_bound(pos_xz)
            if 0 <= p[0] < size and 0 <= p[1] < size:
                x[2, p[1], p[0]] = max(x[2, p[1], p[0]], self.buff_treasures[pos[0], pos[1]])

        # Buff (负值)
        buff_pos = np.argwhere(self.buff_treasures < 0)
        for pos in buff_pos:
            pos_xz = np.array([pos[1], pos[0]], np.int32)
            p = cvt_pos_to_bound(pos_xz)
            if 0 <= p[0] < size and 0 <= p[1] < size:
                x[2, p[1], p[0]] = min(x[2, p[1], p[0]], self.buff_treasures[pos[0], pos[1]])

        # 终点
        end_pos = np.argwhere(self.end > 0)
        for pos in end_pos:
            pos_xz = np.array([pos[1], pos[0]], np.int32)
            p = cvt_pos_to_bound(pos_xz)
            if 0 <= p[0] < size and 0 <= p[1] < size:
                x[3, p[1], p[0]] = max(x[3, p[1], p[0]], self.end[pos[0], pos[1]])

        return x.flatten()

    def _get_map_tensor(self, size=51):
        """获取地图张量 [4, size, size] (用于CNN输入)"""
        assert size % 2 == 1, f"size must be odd, got {size}"

        x = np.zeros((4, size, size), np.float32)

        for i in range(size):
            for j in range(size):
                ii = self.hero_pos[0] + i - size // 2
                jj = self.hero_pos[1] + j - size // 2

                if 0 <= ii < 128 and 0 <= jj < 128:
                    x[0, i, j] = self.obstacles[jj, ii]
                    x[1, i, j] = min(self.memory[jj, ii] / 10.0, 1.0)

        # 处理宝箱/Buff和终点
        def cvt_pos_to_bound(pos):
            center = np.array([size // 2, size // 2], np.int32)
            delta_pos = pos - self.hero_pos

            if abs(delta_pos[0]) <= size // 2 and abs(delta_pos[1]) <= size // 2:
                return delta_pos + center

            theta = math.atan2(delta_pos[1], delta_pos[0])
            if abs(delta_pos[0]) > abs(delta_pos[1]):
                delta_pos[0] = size // 2 * np.sign(delta_pos[0])
                delta_pos[1] = round(size // 2 * abs(math.tan(theta)) * np.sign(delta_pos[1]))
            else:
                delta_pos[1] = size // 2 * np.sign(delta_pos[1])
                delta_pos[0] = round(size // 2 / abs(math.tan(theta)) * np.sign(delta_pos[0]))
            return center + delta_pos.astype(np.int32)

        # 宝箱 (np.argwhere返回[z, x]，需要转换为[x, z])
        treasures_pos = np.argwhere(self.buff_treasures > 0)
        for pos in treasures_pos:
            pos_xz = np.array([pos[1], pos[0]], np.int32)
            p = cvt_pos_to_bound(pos_xz)
            if 0 <= p[0] < size and 0 <= p[1] < size:
                x[2, p[1], p[0]] = max(x[2, p[1], p[0]], self.buff_treasures[pos[0], pos[1]])

        buff_pos = np.argwhere(self.buff_treasures < 0)
        for pos in buff_pos:
            pos_xz = np.array([pos[1], pos[0]], np.int32)
            p = cvt_pos_to_bound(pos_xz)
            if 0 <= p[0] < size and 0 <= p[1] < size:
                x[2, p[1], p[0]] = min(x[2, p[1], p[0]], self.buff_treasures[pos[0], pos[1]])

        end_pos = np.argwhere(self.end > 0)
        for pos in end_pos:
            pos_xz = np.array([pos[1], pos[0]], np.int32)
            p = cvt_pos_to_bound(pos_xz)
            if 0 <= p[0] < size and 0 <= p[1] < size:
                x[3, p[1], p[0]] = max(x[3, p[1], p[0]], self.end[pos[0], pos[1]])

        return x

    def _get_legal_actions(self, legal_act_raw, flash_cooldown):
        """计算合法动作掩码"""
        mask = [True] * Config.ACTION_NUM

        # 根据撞墙历史屏蔽动作
        if self.hit_wall and self.last_action >= 0:
            mask[self.last_action % 8] = False

        # 无闪现时屏蔽闪现动作 (8-15) - cooldown>0表示冷却中，不可用
        if flash_cooldown > 0:
            for i in range(8):
                mask[i + 8] = False

        # 如果全False，则全部开放
        if not any(mask):
            mask = [True] * Config.ACTION_NUM

        return mask

    def _check_hit_wall(self, hero_pos):
        """检测是否撞墙"""
        if self.last_hero_pos is None or self.last_action < 0:
            return False

        last_move_action = self.last_action % 8

        # 检测是否位置没变
        delta_pos = np.abs(hero_pos - self.last_hero_pos)
        delta_distance = np.linalg.norm(delta_pos)

        if delta_distance < 0.1 and self.last_action != -1:
            return True

        return False

    def _build_remain_info(self, env_info, hero_info, hero_pos):
        """构建奖励计算所需的额外信息"""
        # 计算周围记忆
        around_memory = self._get_around_memory()

        # 计算新探索格子数 (简化)
        new_explore = 0

        # 最近宝箱距离
        nearest_treasure_dist = 999
        for i, avail in enumerate(self.avail_treasures):
            if avail and self.last_pos_treasures[i] is not None:
                dist = np.linalg.norm(hero_pos - self.last_pos_treasures[i])
                nearest_treasure_dist = min(nearest_treasure_dist, dist)

        # 最近怪物距离
        nearest_monster_dist = 999
        monsters = env_info.get("monsters", [])
        for m in monsters:
            if m.get("is_in_view", 0):
                m_pos = np.array([m["pos"]["x"], m["pos"]["z"]], np.float32)
                dist = np.linalg.norm(hero_pos - m_pos)
                nearest_monster_dist = min(nearest_monster_dist, dist)

        # 危险等级
        danger_level = 0.0
        if nearest_monster_dist < 30:
            danger_level = 1.0
        elif nearest_monster_dist < 60:
            danger_level = 0.7
        elif nearest_monster_dist < 90:
            danger_level = 0.4
        elif nearest_monster_dist < 120:
            danger_level = 0.2

        # 使用闪现成功逃脱的判断
        escaped = False
        flash_used = self.last_action >= 8 if self.last_action >= 0 else False

        # 从env_info获取flash信息
        flash_cooldown = env_info.get("flash_cooldown", 0)
        flash_status = 1 if flash_cooldown == 0 else 0

        remain_info = {
            "step": self.step,
            "max_step": self.max_step,
            "hero_pos": hero_pos,
            "treasure_collected": self.treasure_collected,
            "buff_count": self.buff_count,
            "flash_status": flash_status,
            "flash_cooldown": flash_cooldown,
            "flash_used": flash_used,
            "hit_wall": self.hit_wall,
            "around_memory": around_memory,
            "new_explore_grid": new_explore,
            "nearest_treasure_dist": nearest_treasure_dist,
            "nearest_treasure_dist_norm": min(nearest_treasure_dist / 180.0, 1.0),
            "nearest_monster_dist": nearest_monster_dist,
            "danger_level": danger_level,
            "escaped": escaped,
            "buff_remain_time": 0,  # 官方文档无此字段，设为0
            # 用于奖励计算的历史信息
            "prev_remain_info": {},  # 由调用者填充
        }

        return remain_info

    def _get_around_memory(self, size=3):
        """获取周围size x size的访问记忆"""
        x = np.zeros((size, size), np.float32)
        for i in range(size):
            for j in range(size):
                ii = self.hero_pos[0] + i - size // 2
                jj = self.hero_pos[1] + j - size // 2
                if 0 <= ii < 128 and 0 <= jj < 128:
                    x[i, j] = self.memory[jj, ii]
        return x
