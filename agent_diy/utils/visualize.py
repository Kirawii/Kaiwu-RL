#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
地图和Agent行为可视化工具
用于诊断agent移动模式、收集行为等
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os


class Visualizer:
    """峡谷追猎可视化器"""

    def __init__(self, save_dir="./visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.trajectory_history = []  # 轨迹历史
        self.reward_history = []  # 奖励历史
        self.step_count = 0

    def reset(self):
        """重置可视化状态"""
        self.trajectory_history = []
        self.reward_history = []
        self.step_count = 0

    def record_step(self, hero_pos, monsters, treasures, buffs, map_tensor, reward, danger_level):
        """
        记录单步状态

        Args:
            hero_pos: (x, z) 英雄位置
            monsters: [(x, z, is_in_view), ...]
            treasures: [(x, z, is_collected), ...]
            buffs: [(x, z, is_collected), ...]
            map_tensor: [4, 51, 51] 地图张量
            reward: float 当前奖励
            danger_level: float 危险等级
        """
        self.trajectory_history.append({
            'hero_pos': hero_pos,
            'monsters': monsters,
            'treasures': treasures,
            'buffs': buffs,
            'map_tensor': map_tensor.copy() if map_tensor is not None else None,
            'reward': reward,
            'danger_level': danger_level,
            'step': self.step_count,
        })
        self.reward_history.append(reward)
        self.step_count += 1

    def plot_trajectory(self, episode_id=0, save_path=None):
        """
        绘制完整轨迹图
        """
        if not self.trajectory_history:
            print("No trajectory data to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        # 提取数据
        hero_xs = [h['hero_pos'][0] for h in self.trajectory_history]
        hero_zs = [h['hero_pos'][1] for h in self.trajectory_history]
        dangers = [h['danger_level'] for h in self.trajectory_history]

        # 获取最新状态的对象位置
        last_state = self.trajectory_history[-1]

        # ===== 子图1: 移动轨迹 =====
        ax1 = axes[0, 0]
        self._plot_map_background(ax1, last_state['map_tensor'])

        # 绘制轨迹（颜色表示危险等级）
        scatter = ax1.scatter(hero_xs, hero_zs, c=dangers, cmap='RdYlGn_r',
                             s=20, alpha=0.6, vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax1, label='Danger Level')

        # 标记起点和终点
        ax1.plot(hero_xs[0], hero_zs[0], 'go', markersize=15, label='Start')
        ax1.plot(hero_xs[-1], hero_zs[-1], 'r*', markersize=20, label='End')

        # 绘制对象
        self._plot_objects(ax1, last_state)

        ax1.set_xlim(0, 128)
        ax1.set_ylim(0, 128)
        ax1.set_title(f'Agent Trajectory (Episode {episode_id})')
        ax1.legend()
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        ax1.grid(True, alpha=0.3)

        # ===== 子图2: 地图通道可视化 =====
        ax2 = axes[0, 1]
        if last_state['map_tensor'] is not None:
            self._plot_map_channels(ax2, last_state['map_tensor'])
        else:
            ax2.text(0.5, 0.5, 'No Map Data', ha='center', va='center')

        # ===== 子图3: 奖励曲线 =====
        ax3 = axes[1, 0]
        steps = list(range(len(self.reward_history)))
        ax3.plot(steps, np.cumsum(self.reward_history), 'b-', label='Cumulative Reward')
        ax3.plot(steps, self.reward_history, 'r-', alpha=0.3, label='Step Reward')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Reward')
        ax3.set_title('Reward Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # ===== 子图4: 统计信息 =====
        ax4 = axes[1, 1]
        self._plot_statistics(ax4)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, f'episode_{episode_id:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.close()

    def _plot_map_background(self, ax, map_tensor):
        """绘制地图背景（障碍物）"""
        if map_tensor is None or map_tensor.shape != (4, 51, 51):
            return

        # 使用通道0（障碍物）作为背景
        obstacles = map_tensor[0]

        # 创建背景图
        bg = np.ones((51, 51, 3))  # 白色背景
        bg[obstacles < 0] = [0.8, 0.8, 0.9]  # 未知区域灰色
        bg[obstacles == 0] = [0.3, 0.3, 0.3]  # 障碍物深灰

        # 注意：map_tensor是局部视图，需要映射到全局坐标
        ax.imshow(bg, extent=[0, 128, 0, 128], origin='lower', alpha=0.3)

    def _plot_objects(self, ax, state):
        """绘制宝箱、Buff、怪物"""
        # 宝箱
        for tx, tz, collected in state['treasures']:
            if not collected:
                circle = patches.Circle((tx, tz), 3, fill=True,
                                       facecolor='gold', edgecolor='orange',
                                       linewidth=2, alpha=0.7, label='Treasure')
                ax.add_patch(circle)

        # Buff
        for bx, bz, collected in state['buffs']:
            if not collected:
                circle = patches.Circle((bx, bz), 3, fill=True,
                                       facecolor='cyan', edgecolor='blue',
                                       linewidth=2, alpha=0.7, label='Buff')
                ax.add_patch(circle)

        # 怪物
        for mx, mz, in_view in state['monsters']:
            color = 'red' if in_view else 'pink'
            circle = patches.Circle((mx, mz), 5, fill=True,
                                   facecolor=color, edgecolor='darkred',
                                   linewidth=2, alpha=0.7, label='Monster')
            ax.add_patch(circle)

    def _plot_map_channels(self, ax, map_tensor):
        """绘制4个地图通道"""
        titles = ['Obstacles', 'Visit Memory', 'Treasure/Buff', 'Monsters']
        cmaps = ['gray', 'hot', 'RdYlGn', 'Reds']

        for i in range(4):
            ax_sub = plt.subplot(2, 2, i+1)
            data = map_tensor[i]

            if i == 2:  # Treasure/Buff通道，正负区分
                im = ax_sub.imshow(data, cmap='RdYlGn', vmin=-1, vmax=1, origin='lower')
            else:
                im = ax_sub.imshow(data, cmap=cmaps[i], origin='lower')

            plt.colorbar(im, ax=ax_sub)
            ax_sub.set_title(titles[i])
            ax_sub.axis('off')

    def _plot_statistics(self, ax):
        """绘制统计信息"""
        stats_text = []

        # 基本统计
        stats_text.append(f"Total Steps: {len(self.trajectory_history)}")
        stats_text.append(f"Total Reward: {sum(self.reward_history):.2f}")

        # 移动距离
        if len(self.trajectory_history) > 1:
            total_dist = 0
            for i in range(1, len(self.trajectory_history)):
                p1 = self.trajectory_history[i-1]['hero_pos']
                p2 = self.trajectory_history[i]['hero_pos']
                dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                total_dist += dist
            stats_text.append(f"Total Distance: {total_dist:.1f}")

        # 危险统计
        danger_steps = sum(1 for h in self.trajectory_history if h['danger_level'] > 0.5)
        stats_text.append(f"Danger Steps: {danger_steps} ({danger_steps/len(self.trajectory_history)*100:.1f}%)")

        # 收集统计
        treasures_collected = sum(1 for h in self.trajectory_history
                                 for t in h['treasures'] if t[2])
        buffs_collected = sum(1 for h in self.trajectory_history
                             for b in h['buffs'] if b[2])
        stats_text.append(f"Treasures Collected: {treasures_collected}")
        stats_text.append(f"Buffs Collected: {buffs_collected}")

        ax.text(0.1, 0.9, '\n'.join(stats_text), transform=ax.transAxes,
               fontsize=12, verticalalignment='top', family='monospace')
        ax.axis('off')
        ax.set_title('Statistics')

    def create_animation(self, episode_id=0, save_path=None, interval=100):
        """
        创建轨迹动画
        """
        if not self.trajectory_history:
            print("No trajectory data to animate")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # 设置坐标范围
        ax.set_xlim(0, 128)
        ax.set_ylim(0, 128)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title(f'Agent Movement Animation (Episode {episode_id})')
        ax.grid(True, alpha=0.3)

        # 初始化绘制元素
        hero_dot, = ax.plot([], [], 'bo', markersize=10, label='Hero')
        monster_dots = []
        treasure_dots = []
        buff_dots = []
        trajectory_line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)

        # 静态对象（只初始化一次）
        last_state = self.trajectory_history[-1]
        for tx, tz, collected in last_state['treasures']:
            if not collected:
                ax.plot(tx, tz, 'yo', markersize=8, alpha=0.5)
        for bx, bz, collected in last_state['buffs']:
            if not collected:
                ax.plot(bx, bz, 'co', markersize=8, alpha=0.5)
        for mx, mz, in_view in last_state['monsters']:
            color = 'ro' if in_view else 'mo'
            ax.plot(mx, mz, color, markersize=10, alpha=0.5)

        def init():
            hero_dot.set_data([], [])
            trajectory_line.set_data([], [])
            return hero_dot, trajectory_line

        def update(frame):
            state = self.trajectory_history[frame]

            # 更新英雄位置
            hx, hz = state['hero_pos']
            hero_dot.set_data([hx], [hz])

            # 更新轨迹线
            xs = [h['hero_pos'][0] for h in self.trajectory_history[:frame+1]]
            zs = [h['hero_pos'][1] for h in self.trajectory_history[:frame+1]]
            trajectory_line.set_data(xs, zs)

            # 更新标题显示步数和危险等级
            ax.set_title(f'Episode {episode_id} - Step {frame} - Danger: {state[\"danger_level\"]:.2f}')

            return hero_dot, trajectory_line

        anim = FuncAnimation(fig, update, frames=len(self.trajectory_history),
                           init_func=init, blit=True, interval=interval)

        if save_path is None:
            save_path = os.path.join(self.save_dir, f'episode_{episode_id:04d}_animation.mp4')

        try:
            anim.save(save_path, writer='ffmpeg', fps=10)
            print(f"Saved animation to {save_path}")
        except Exception as e:
            print(f"Failed to save animation: {e}")
            # 尝试保存为gif
            gif_path = save_path.replace('.mp4', '.gif')
            try:
                anim.save(gif_path, writer='pillow', fps=10)
                print(f"Saved animation as GIF to {gif_path}")
            except Exception as e2:
                print(f"Failed to save GIF: {e2}")

        plt.close()


class PreprocessorWithVis:
    """带可视化的预处理器包装"""

    def __init__(self, preprocessor, visualizer=None, enable_vis=True, vis_every_n_episodes=1):
        self.preprocessor = preprocessor
        self.visualizer = visualizer or Visualizer()
        self.enable_vis = enable_vis
        self.vis_every_n_episodes = vis_every_n_episodes
        self.episode_count = 0
        self.current_episode_data = []

    def reset(self):
        """每局开始时调用"""
        if self.enable_vis and self.current_episode_data:
            # 保存上一局的可视化
            if self.episode_count % self.vis_every_n_episodes == 0:
                self.visualizer.trajectory_history = self.current_episode_data
                self.visualizer.plot_trajectory(self.episode_count)
            self.episode_count += 1

        self.current_episode_data = []
        return self.preprocessor.reset()

    def feature_process(self, env_obs, last_action):
        """处理特征并记录可视化数据"""
        result = self.preprocessor.feature_process(env_obs, last_action)

        if self.enable_vis:
            # 提取需要的数据
            obs_data, remain_info = result[:2]

            # 从 env_obs 提取位置信息
            observation = env_obs.get("observation", {})
            frame_state = observation.get("frame_state", {})
            hero = frame_state.get("heroes", {})
            hero_pos = (hero.get("pos", {}).get("x", 0), hero.get("pos", {}).get("z", 0))

            # 提取怪物、宝箱、Buff信息
            monsters = []
            for m in frame_state.get("monsters", []):
                pos = m.get("pos", {})
                monsters.append((pos.get("x", 0), pos.get("z", 0), m.get("is_in_view", 0)))

            treasures = []
            buffs = []
            for organ in frame_state.get("organs", []):
                pos = organ.get("pos", {})
                x, z = pos.get("x", 0), pos.get("z", 0)
                if organ.get("sub_type") == 1:  # 宝箱
                    collected = organ.get("status", 1) == 0
                    treasures.append((x, z, collected))
                elif organ.get("sub_type") == 2:  # Buff
                    collected = organ.get("status", 1) == 0
                    buffs.append((x, z, collected))

            # 记录
            self.current_episode_data.append({
                'hero_pos': hero_pos,
                'monsters': monsters,
                'treasures': treasures,
                'buffs': buffs,
                'map_tensor': getattr(obs_data, 'map_tensor', None),
                'reward': remain_info.get('last_reward', 0),
                'danger_level': remain_info.get('danger_level', 0),
            })

        return result


if __name__ == '__main__':
    # 测试可视化器
    print("Testing visualizer...")
    vis = Visualizer()

    # 模拟一些数据
    for step in range(100):
        hero_pos = (64 + step * 0.5, 64 + np.sin(step * 0.1) * 10)
        monsters = [(80, 80, True)]
        treasures = [(70, 70, False), (50, 50, step > 50)]
        buffs = [(60, 60, step > 30)]
        map_tensor = np.random.randn(4, 51, 51)
        reward = np.random.randn()
        danger = np.random.random()

        vis.record_step(hero_pos, monsters, treasures, buffs, map_tensor, reward, danger)

    vis.plot_trajectory(episode_id=0)
    print("Test completed!")
