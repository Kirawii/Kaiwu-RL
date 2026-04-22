# Agent DIY - 峡谷追猎强化学习智能体 (优化版本)

## 概述

本实现是腾讯开悟平台"峡谷追猎"比赛的完整 DIY Agent 方案，**参考夺冠队伍经验进行全面优化**。

## 核心改进 (基于readme夺冠经验)

### 1. 动作掩码损失 (Action Mask Loss) ⭐
**来源**: readme 4.1节 - 解决"站着不动"问题的关键技巧

```python
# 危险时不使用闪现 -> 惩罚
# 安全时使用闪现 -> 惩罚 (浪费)
action_mask_loss = danger * (1 - flash_used) + safe * flash_used
```

**效果**: 1小时内解决无效行为，实现闪现的精准释放

### 2. 课程学习 (Curriculum Learning) ⭐
**来源**: readme 5.1节和7.1节

| 阶段 | 比例 | 策略重点 | 宝箱奖励 | 生存权重 |
|------|------|----------|----------|----------|
| 前期 | 30% | 探索收集 | 15.0 | 低 |
| 中期 | 40% | 平衡 | 10.0 | 中 |
| 后期 | 30% | 保守生存 | 5.0 | 高 |

### 3. 情景奖励 (Situational Reward) ⭐
**来源**: readme 6节 - 核心技巧，直接提高2个baseline

```python
# 情景1: 危险时冒险收集宝箱 -> 惩罚 (-3.0)
if danger > 0.7 and closer_to_treasure:
    reward -= 3.0

# 情景2: 近距离成功逃脱 -> 大奖励 (+1.0)
if escaped and monster_dist < 40:
    reward += 1.0

# 情景3: 浪费闪现 -> 惩罚 (-1.0)
if flash_used and danger < 0.2:
    reward -= 1.0
```

### 4. FC + GRU 混合网络 ⭐
**来源**: readme 5.1节 - FC:GRU = 3:1，防止过拟合

```
输入 (96D)
  ↓
FC骨干 (3层，192维) → ReLU → PSCN模块
  ↓
GRU (1层，64维)  [捕捉时序]
  ↓
拼接 (256D) → Fusion (128D)
  ↓
Actor/Critic 双头
```

**PSCN模块**: Power Series Connected Network，类似残差但参数更少

### 5. 动态超参数 ⭐
**来源**: readme 7.1节调参建议

| 参数 | 初始值 | 终值 | 变化方式 |
|------|--------|------|----------|
| Gamma | 0.995 | 0.9975 | 渐增 |
| Lambda | 0.95 | 0.98 | 渐增 |
| Beta (熵) | 0.015 | 0.001 | 渐减 |
| LR | 3e-4 | 3e-6 | 指数衰减 |

### 6. Dual-Clip PPO
**来源**: readme 4节提到的dual-clip

```python
# 标准PPO: ratio in [1-ε, 1+ε]
# Dual-Clip: 增加上限 3.0，防止ratio过大
surr3 = 3.0 * advantage
policy_loss = -min(max(surr1, surr2), surr3).mean()
```

### 7. 动作空间扩展 (16维)
| 动作索引 | 类型 | 方向 | 距离 |
|---------|------|------|------|
| 0-7 | 移动 | 8方向 | 1格/步 |
| 8-15 | 闪现 | 8方向 | 正交10格/斜向8格 |

### 8. 多头价值估计
```python
value = [生存价值, 宝箱价值]
advantage = 0.7 * survive_advantage + 0.3 * treasure_advantage
```

## 特征工程 (96维)

| 特征类别 | 维度 | 说明 |
|---------|------|------|
| 英雄自身 | 6D | 位置、闪现CD、闪现可用、Buff剩余、加速状态 |
| 怪物特征 | 12D | 2只 × (可见性、位置、速度、距离桶、方向) |
| 宝箱特征 | 60D | 10个 × (激活状态、位置、距离桶、方向、优先级) |
| Buff特征 | 12D | 2个 × (激活状态、位置、距离桶、方向、剩余时间) |
| 局部地图 | 25D | 5×5区域通行性 |
| 进度特征 | 4D | 步数、生存率、怪物2计时、**危险等级** |
| 合法动作 | 16D | 动作掩码 |

## 奖励设计 (密集化 + 情景化)

### 基础奖励
| 奖励项 | 数值 | 说明 |
|--------|------|------|
| 生存基础 | +0.01/步 | 每步生存奖励 |
| 距离塑形 | ±0.05×Δdist | 远离怪物正奖励 |
| 步数奖励 | +0.015×progress | 活得越久越高 |

### 宝箱奖励 (课程学习)
| 阶段 | 收集奖励 | 探索奖励 |
|------|----------|----------|
| 前期 | +15.0/个 | +0.001/步 |
| 中期 | +10.0/个 | +0.001/步 |
| 后期 | +5.0/个 | +0.0005/步 |

### 情景奖励 (核心)
| 情景 | 奖励 | 说明 |
|------|------|------|
| 危险时收集 | -3.0 | 惩罚冒险行为 |
| 安全时不探索 | -0.1 | 轻微惩罚保守 |
| 近距离逃脱 | +1.0 | 鼓励精彩操作 |
| 闪现逃脱 | +2.0 | 技能使用奖励 |
| 浪费闪现 | -1.0 | 惩罚乱用技能 |
| 通关 | +10.0 | 存活奖励 |
| 死亡 | -10.0 | 死亡惩罚 |

## 网络结构详情

```
Model: "gorge_chase_diy_v2"

Feature Extractor:
  - FC Layer 1: 96 -> 192, ReLU
  - PSCN Module: 192 -> 96 -> 192, Residual
  - FC Layer 2: 192 -> 192, ReLU
  - GRU: 192 -> 64 (1 layer)
  - Fusion: concat(192, 64) -> 256 -> 128, ReLU

Actor Head:
  - Hidden: 128 -> 64, ReLU
  - Output: 64 -> 16 (actions)

Critic Head:
  - Hidden: 128 -> 64, ReLU
  - Output: 64 -> 2 (multi-value)

Auxiliary Task:
  - Danger Predictor: 128 -> 32 -> 1, Sigmoid
```

## 训练流程

```python
# 每局游戏
1. Reset Agent (重置GRU状态)
2. Load latest model
3. While not done:
   a. observation_process() -> feature (96D)
   b. predict() -> action (sample from policy)
   c. env.step(action) -> next_obs, reward
   d. reward_shaping() -> shaped_reward (课程+情景)
   e. collect sample
4. sample_process() -> GAE with dynamic gamma/lambda
5. learn() -> PPO update with action mask loss
6. Report metrics
```

## 监控指标

### 算法指标
- `reward_survive`: 生存奖励
- `reward_treasure`: 宝箱奖励
- `total_loss`: 总损失
- `policy_loss`: 策略损失
- `value_loss`: 价值损失
- `entropy`: 策略熵
- `action_mask_loss`: 动作掩码损失
- `beta`: 当前熵系数

### 环境指标
- `total_score`: 总得分
- `treasures_collected`: 宝箱收集数
- `episode_steps`: 对局步数
- `escape_count`: 成功逃脱次数

## 文件结构

```
agent_diy/
├── agent.py                 # Agent主类 (GRU状态管理)
├── algorithm/
│   └── algorithm.py         # PPO算法 (Action Mask + Dual-Clip)
├── conf/
│   ├── conf.py              # 配置 (动态参数)
│   ├── monitor_builder.py   # 监控配置
│   └── train_env_conf.toml  # 环境配置
├── feature/
│   ├── definition.py        # 数据结构、GAE、动态参数
│   └── preprocessor.py      # 特征工程 (课程+情景)
├── model/
│   └── model.py             # FC+GRU混合网络 + PSCN
└── workflow/
    └── train_workflow.py    # 训练流程
```

## 使用方法

```bash
# 确保使用diy算法
# train_test.py 中: algorithm_name = "diy"

python train_test.py
```

## 优化效果预期

基于readme夺冠经验，本实现预期带来以下提升：

| 改进项 | 预期提升 | 依据 |
|--------|----------|------|
| 动作掩码损失 | 基础行为稳定 | 1小时解决"站着不动" |
| 情景奖励 | +2 baseline | readme明确说明 |
| FC+GRU混合 | 防过拟合 | 夺冠队伍验证 |
| 课程学习 | 后期生存率↑ | 动态调整策略 |
| 动态超参 | 收敛稳定性↑ | 专业调参经验 |
| 闪现策略 | 逃脱成功率↑ | 精准释放 |

## 后续优化方向

1. **Self-Play**: 使用历史模型作为对手 (readme 4.2节)
2. **模型融合**: Module Soup技术 (readme 4.4节)
3. **注意力机制**: 动态关注重要宝箱 (readme 5.4节)
4. **多智能体**: 扩展到3v3场景 (readme 4.3节)

## 参考

- 腾讯开悟平台文档
- 夺冠队伍readme经验分享
- PPO: Proximal Policy Optimization Algorithms (Schulman et al.)
- GAE: High-Dimensional Continuous Control Using Generalized Advantage Estimation
