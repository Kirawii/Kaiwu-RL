#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
调试工具 - 简化版可视化
无需matplotlib，用ASCII字符在控制台输出
"""

import numpy as np
import os


class SimpleMapVisualizer:
    """简单的ASCII地图可视化器"""

    def __init__(self, map_size=128, view_size=51):
        self.map_size = map_size
        self.view_size = view_size
        self.half_view = view_size // 2
        self.reset()

    def reset(self):
        """重置状态"""
        self.hero_trajectory = []  # 英雄轨迹
        self.treasure_positions = set()  # 宝箱位置
        self.buff_positions = set()  # Buff位置
        self.monster_trajectories = [[], []]  # 两个怪物的轨迹
        self.collected_treasures = set()
        self.collected_buffs = set()

    def record(self, hero_pos, monsters, organs, step):
        """
        记录一帧数据

        Args:
            hero_pos: {'x', 'z'}
            monsters: list of {'pos': {'x', 'z'}, 'is_in_view'}
            organs: list of {'sub_type', 'pos': {'x', 'z'}, 'status'}
            step: int
        """
        hx, hz = int(hero_pos['x']), int(hero_pos['z'])
        self.hero_trajectory.append((hx, hz, step))

        # 记录怪物位置
        for i, m in enumerate(monsters[:2]):
            if m.get('is_in_view', 0):
                mx = int(m['pos']['x'])
                mz = int(m['pos']['z'])
                self.monster_trajectories[i].append((mx, mz, step))

        # 记录宝箱和Buff位置
        for organ in organs:
            sub_type = organ.get('sub_type')
            pos = organ.get('pos', {})
            x, z = int(pos.get('x', 0)), int(pos.get('z', 0))
            status = organ.get('status', 1)

            if sub_type == 1:  # 宝箱
                if status == 1:  # 未收集
                    self.treasure_positions.add((x, z))
                else:  # 已收集
                    self.collected_treasures.add((x, z))
                    self.treasure_positions.discard((x, z))
            elif sub_type == 2:  # Buff
                if status == 1:
                    self.buff_positions.add((x, z))
                else:
                    self.collected_buffs.add((x, z))
                    self.buff_positions.discard((x, z))

    def render(self, hero_pos=None, step=None, danger_level=0):
        """
        渲染当前状态到控制台

        图例:
        H = 英雄
        T = 宝箱
        B = Buff
        M = 怪物
        * = 英雄轨迹
        . = 空地
        # = 障碍物（如果可见）
        """
        if hero_pos is None:
            if not self.hero_trajectory:
                return "No data"
            hero_pos = {'x': self.hero_trajectory[-1][0], 'z': self.hero_trajectory[-1][1]}

        hx, hz = int(hero_pos['x']), int(hero_pos['z'])

        # 确定显示范围（以英雄为中心）
        x_min = max(0, hx - self.half_view)
        x_max = min(self.map_size, hx + self.half_view)
        z_min = max(0, hz - self.half_view)
        z_max = min(self.map_size, hz + self.half_view)

        # 创建地图网格
        width = x_max - x_min
        height = z_max - z_min

        # 限制显示大小
        if width > 51:
            x_min = hx - 25
            x_max = hx + 26
            width = 51
        if height > 51:
            z_min = hz - 25
            z_max = hz + 26
            height = 51

        grid = [['.' for _ in range(width)] for _ in range(height)]

        # 绘制轨迹（排除当前位置）
        for x, z, s in self.hero_trajectory[:-1]:
            if x_min <= x < x_max and z_min <= z < z_max:
                gx, gz = x - x_min, z - z_min
                if grid[height - 1 - gz][gx] == '.':  # 不覆盖其他标记
                    grid[height - 1 - gz][gx] = '*'

        # 绘制已收集的物品（低优先级）
        for x, z in self.collected_treasures:
            if x_min <= x < x_max and z_min <= z < z_max:
                gx, gz = x - x_min, z - z_min
                grid[height - 1 - gz][gx] = 't'

        for x, z in self.collected_buffs:
            if x_min <= x < x_max and z_min <= z < z_max:
                gx, gz = x - x_min, z - z_min
                grid[height - 1 - gz][gx] = 'b'

        # 绘制未收集的宝箱和Buff
        for x, z in self.treasure_positions:
            if x_min <= x < x_max and z_min <= z < z_max:
                gx, gz = x - x_min, z - z_min
                grid[height - 1 - gz][gx] = 'T'

        for x, z in self.buff_positions:
            if x_min <= x < x_max and z_min <= z < z_max:
                gx, gz = x - x_min, z - z_min
                grid[height - 1 - gz][gx] = 'B'

        # 绘制怪物轨迹
        for i, traj in enumerate(self.monster_trajectories):
            for x, z, s in traj[:-1]:
                if x_min <= x < x_max and z_min <= z < z_max:
                    gx, gz = x - x_min, z - z_min
                    if grid[height - 1 - gz][gx] == '.':
                        grid[height - 1 - gz][gx] = '+'

        # 绘制当前怪物位置
        for i, traj in enumerate(self.monster_trajectories):
            if traj:
                x, z, s = traj[-1]
                if x_min <= x < x_max and z_min <= z < z_max:
                    gx, gz = x - x_min, z - z_min
                    grid[height - 1 - gz][gx] = 'M'

        # 绘制英雄位置（最高优先级）
        if x_min <= hx < x_max and z_min <= hz < z_max:
            gx, gz = hx - x_min, hz - z_min
            grid[height - 1 - gz][gx] = 'H'

        # 生成输出字符串
        lines = []
        danger_str = "HIGH" if danger_level > 0.7 else ("MID" if danger_level > 0.3 else "LOW")
        lines.append(f"=== Step {step} | Danger: {danger_str} ({danger_level:.2f}) | Hero: ({hx}, {hz}) ===")
        lines.append(f"  Collected: {len(self.collected_treasures)}T {len(self.collected_buffs)}B | Remaining: {len(self.treasure_positions)}T {len(self.buff_positions)}B")
        lines.append("-" * (width + 2))

        for row in grid:
            lines.append("|" + "".join(row) + "|")

        lines.append("-" * (width + 2))
        lines.append("Legend: H=Hero T=Treasure B=Buff M=Monster *=Path t/b=Collected")

        return "\n".join(lines)

    def save_to_file(self, filepath, hero_pos=None, step=None, danger_level=0):
        """保存渲染结果到文件"""
        output = self.render(hero_pos, step, danger_level)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        return output


class EpisodeLogger:
    """对局日志记录器"""

    def __init__(self, log_dir="./episode_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_count = 0
        self.reset()

    def reset(self):
        """重置当前对局记录"""
        self.step_records = []
        self.episode_count += 1
        self.current_step = 0
        self.visualizer = SimpleMapVisualizer()

    def log_step(self, env_obs, reward, action, remain_info):
        """
        记录单步数据

        Returns:
            debug_info: dict 可用于调试的信息
        """
        observation = env_obs.get("observation", {})
        frame_state = observation.get("frame_state", {})
        hero = frame_state.get("heroes", {})
        hero_pos = hero.get("pos", {})
        monsters = frame_state.get("monsters", [])
        organs = frame_state.get("organs", [])

        # 记录到可视化器
        self.visualizer.record(hero_pos, monsters, organs, self.current_step)

        # 记录关键信息
        record = {
            'step': self.current_step,
            'hero_pos': hero_pos,
            'monsters': [{'pos': m.get('pos'), 'is_in_view': m.get('is_in_view')} for m in monsters],
            'treasure_count': remain_info.get('treasure_collected', 0),
            'buff_count': remain_info.get('buff_collected', 0),
            'danger_level': remain_info.get('danger_level', 0),
            'reward': float(reward) if hasattr(reward, '__float__') else 0,
            'action': action,
            'is_early': remain_info.get('is_early_phase', False),
            'is_late': remain_info.get('is_late_phase', False),
        }
        self.step_records.append(record)
        self.current_step += 1

        # 返回调试信息
        return {
            'step': self.current_step,
            'hero': f"({hero_pos.get('x', 0):.0f}, {hero_pos.get('z', 0):.0f})",
            'treasures': record['treasure_count'],
            'buffs': record['buff_count'],
            'danger': f"{record['danger_level']:.2f}",
            'reward': f"{record['reward']:.3f}",
        }

    def save_episode(self, filename=None):
        """保存对局记录到文件"""
        if filename is None:
            filename = f"episode_{self.episode_count:04d}.txt"

        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            # 写入最终地图可视化
            if self.step_records:
                last_record = self.step_records[-1]
                f.write("FINAL MAP STATE:\n")
                map_render = self.visualizer.render(
                    hero_pos=last_record['hero_pos'],
                    step=last_record['step'],
                    danger_level=last_record['danger_level']
                )
                f.write(map_render)
                f.write("\n\n")

            # 写入步数记录
            f.write("STEP DETAILS:\n")
            f.write(f"{'Step':<6} {'Hero Pos':<15} {'T':<3} {'B':<3} {'Danger':<8} {'Reward':<10} {'Action':<6}\n")
            f.write("-" * 60 + "\n")

            for r in self.step_records:
                pos = r['hero_pos']
                pos_str = f"({pos.get('x', 0):.0f}, {pos.get('z', 0):.0f})"
                f.write(f"{r['step']:<6} {pos_str:<15} {r['treasure_count']:<3} {r['buff_count']:<3} "
                       f"{r['danger_level']:<8.2f} {r['reward']:<10.3f} {r['action']:<6}\n")

            # 写入统计信息
            f.write("\n" + "=" * 60 + "\n")
            f.write("STATISTICS:\n")
            if self.step_records:
                total_reward = sum(r['reward'] for r in self.step_records)
                avg_danger = sum(r['danger_level'] for r in self.step_records) / len(self.step_records)
                final_treasures = self.step_records[-1]['treasure_count']
                final_buffs = self.step_records[-1]['buff_count']

                f.write(f"Total Steps: {len(self.step_records)}\n")
                f.write(f"Total Reward: {total_reward:.3f}\n")
                f.write(f"Avg Danger: {avg_danger:.3f}\n")
                f.write(f"Final Treasures: {final_treasures}\n")
                f.write(f"Final Buffs: {final_buffs}\n")

        print(f"Episode log saved to {filepath}")
        return filepath


def test_visualizer():
    """测试可视化器"""
    vis = SimpleMapVisualizer()

    # 模拟数据
    for step in range(20):
        hero_pos = {'x': 60 + step * 2, 'z': 60 + (step % 5) * 3}
        monsters = [
            {'pos': {'x': 80, 'z': 80}, 'is_in_view': 1},
        ]
        organs = [
            {'sub_type': 1, 'pos': {'x': 70, 'z': 70}, 'status': 1 if step < 5 else 0},
            {'sub_type': 1, 'pos': {'x': 90, 'z': 50}, 'status': 1},
            {'sub_type': 2, 'pos': {'x': 50, 'z': 60}, 'status': 1 if step < 10 else 0},
        ]

        vis.record(hero_pos, monsters, organs, step)

    print(vis.render(hero_pos, step=20, danger_level=0.5))


if __name__ == '__main__':
    test_visualizer()
