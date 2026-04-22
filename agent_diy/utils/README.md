# 可视化调试工具使用说明

## 简介

为了帮助诊断agent行为问题（如宝箱收集率低、Buff从不收集等），我们提供了可视化调试工具。

## 工具列表

### 1. SimpleMapVisualizer (简单地图可视化器)
- ASCII字符地图，无需额外依赖
- 实时显示agent轨迹、宝箱、Buff、怪物位置
- 适合快速调试和远程服务器使用

### 2. EpisodeLogger (对局日志记录器)
- 记录每步的详细状态
- 生成文本格式的地图和统计数据
- 保存到文件供后续分析

### 3. Visualizer (高级可视化器)
- 需要matplotlib
- 生成PNG图片和MP4动画
- 适合深度分析

## 使用方法

### 启用调试模式

在启动训练前设置环境变量：

```bash
export ENABLE_DEBUG=true
python train.py
```

或者在代码中修改：

```python
# agent_diy/workflow/train_workflow.py
enable_debug = True  # 第56行
```

### 查看输出

调试信息会自动保存到 `./episode_logs/` 目录：

```
episode_logs/
├── episode_0001.txt    # 第1局详细记录
├── episode_0002.txt
├── episode_0010.txt
└── episode_0100.txt    # 每100局保存一次
```

### 日志内容示例

```
FINAL MAP STATE:
=== Step 500 | Danger: LOW (0.15) | Hero: (64, 64) ===
  Collected: 7T 1B | Remaining: 3T 1B
----------------------------------------------------
|..................................................|
|....................T.............................|
|...*.........*.........*.........*................|
|..................................................|
|.......B..........................................|
|..................................................|
|.*.........H.........*.........*.........*........|
|..................................................|
|.........M........................................|
----------------------------------------------------
Legend: H=Hero T=Treasure B=Buff M=Monster *=Path t/b=Collected

STEP DETAILS:
Step   Hero Pos        T   B   Danger   Reward     Action
------------------------------------------------------------
0      (64, 64)        0   0   0.00     0.000      0
50     (70, 65)        1   0   0.20     15.230     1
100    (55, 70)        2   0   0.35     5.120      3
...

STATISTICS:
Total Steps: 500
Total Reward: 156.340
Avg Danger: 0.25
Final Treasures: 7
Final Buffs: 1
```

## 图例说明

| 符号 | 含义 |
|------|------|
| H | 英雄当前位置 |
| T | 未收集的宝箱 |
| B | 未收集的Buff |
| M | 怪物位置 |
| * | 英雄走过的路径 |
| t | 已收集的宝箱 |
| b | 已收集的Buff |
| . | 空地 |

## 常见问题诊断

### 问题1: 宝箱收集率低
**检查点:**
- 日志中 `T` 是否出现在 `H` 附近？
- `*` 轨迹是否避开了 `T`？
- `Collected: xT` 中的数字是否增加？

**可能原因:**
- 奖励函数中宝箱奖励权重太低
- Agent过于保守，优先逃跑而非收集
- 地图特征没有正确传递给模型

### 问题2: 从不收集Buff
**检查点:**
- 日志中是否出现 `B`？
- 英雄是否接近过 `B`？
- `Collected: xB` 是否始终为0？

**可能原因:**
- Buff奖励太低 (`REW_BUFF` 只有0.5)
- Buff特征在preprocessor中丢失
- Agent看不到Buff（is_in_view问题）

### 问题3: 原地转圈/重复路径
**检查点:**
- `*` 是否在同一区域密集？
- 是否出现来回往复的路径？

**可能原因:**
- 探索奖励设置不当
- 记忆惩罚机制问题
- 价值函数估计不准

## 高级用法

### 手动调用可视化器

```python
from agent_diy.utils.debug_utils import SimpleMapVisualizer

vis = SimpleMapVisualizer()

# 在游戏循环中记录
for step in range(1000):
    # ... 游戏逻辑 ...
    vis.record(hero_pos, monsters, organs, step)

# 渲染最终状态
print(vis.render(hero_pos, step, danger_level))
```

### 生成Matplotlib图表

```python
from agent_diy.utils.visualize import Visualizer

vis = Visualizer(save_dir="./visualizations")

# 记录每步数据
vis.record_step(hero_pos, monsters, treasures, buffs, map_tensor, reward, danger)

# 生成轨迹图
vis.plot_trajectory(episode_id=1)

# 生成动画（需要ffmpeg）
vis.create_animation(episode_id=1)
```

## 依赖安装

基础功能（ASCII可视化）无需额外依赖。

高级可视化需要：

```bash
pip install matplotlib

# 如需生成MP4动画，还需安装ffmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Mac:
brew install ffmpeg

# Windows:
# 下载ffmpeg并添加到PATH
```

## 性能注意

- 调试模式会增加内存使用和IO开销
- 建议只在诊断问题时启用
- 生产环境训练应关闭调试模式
