# 峡谷追猎强化学习智能体设计方案

## 汇报信息
- **项目名称**: 腾讯开悟平台 - 峡谷追猎 (Gorge Chase) AI 智能体
- **方案版本**: DIY Agent v2.0
- **参考依据**: 第二届开悟比赛冠军队伍技术分享
- **文档日期**: 2026年4月7日

---

## 一、项目背景

### 1.1 赛题介绍
峡谷追猎是腾讯开悟平台推出的路径规划+生存博弈类强化学习赛题：

| 要素 | 说明 |
|------|------|
| **目标** | 在1000步内存活，同时收集尽可能多的宝箱 |
| **角色** | 鲁班七号（玩家）vs 2个怪物 |
| **地图** | 128×128栅格地图，共15张（10公开+5隐藏）|
| **技能** | 超级闪现：正交10格/斜向8格穿越 |
| **增益** | 加速buff：移速翻倍 |

### 1.2 核心挑战
1. **样本效率**: 训练资源有限（1024 core），需要高效利用样本
2. **多目标权衡**: 生存 vs 收集宝箱的权衡
3. **技能使用**: 闪现的精准使用时机
4. **泛化能力**: 需要在未见过的地图上表现良好

---

## 二、方案概述

本方案基于**夺冠队伍的技术分享（readme.md）**进行优化，实现了以下7大核心改进：

```
┌─────────────────────────────────────────────────────────────┐
│                    DIY Agent v2.0 架构                       │
├─────────────────────────────────────────────────────────────┤
│  特征工程 (135维)                                            │
│  ├── 英雄特征 (6D): 位置、技能CD、Buff状态                    │
│  ├── 怪物特征 (12D): 2只怪物的位置、速度、距离、方向          │
│  ├── 宝箱特征 (60D): 10个宝箱的位置、距离、优先级             │
│  ├── Buff特征 (12D): 2个加速buff的状态                       │
│  ├── 地图特征 (25D): 5×5局部通行性                           │
│  ├── 进度特征 (4D): 步数、生存率、怪物计时、危险等级          │
│  └── 动作掩码 (16D): 合法动作指示                            │
├─────────────────────────────────────────────────────────────┤
│  网络结构 (FC+GRU混合，3:1比例)                              │
│  ├── FC骨干 (192维×3层): 主要特征提取                         │
│  ├── PSCN模块: 类残差结构，减少参数冗余                       │
│  ├── GRU (64维×1层): 时序信息，防止过拟合                     │
│  ├── Actor头: 16维动作概率                                    │
│  └── Critic头: 2维多头价值 (生存+宝箱)                        │
├─────────────────────────────────────────────────────────────┤
│  算法优化 (PPO + Dual-Clip + 动作掩码)                        │
│  ├── Dual-Clip PPO: 更稳定的策略更新                          │
│  ├── 动作掩码损失: 规范闪现使用行为                            │
│  ├── 多头价值: 分别估计生存和收集价值                          │
│  └── 动态超参: Gamma/Lambda/Beta渐进调整                      │
├─────────────────────────────────────────────────────────────┤
│  奖励设计 (课程学习 + 情景奖励)                               │
│  ├── 课程学习: 前期重收集，后期重生存                          │
│  ├── 情景奖励: 危险时收集惩罚、逃脱奖励等                      │
│  └── 密集奖励: 距离塑形、探索奖励、步数奖励                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、核心改进详解

### 3.1 动作掩码损失 (Action Mask Loss)
**参考**: readme 4.1节

**问题**: 基线版本存在"站着不动"、"乱交闪现"等行为问题

**解决方案**:
```python
# 危险时不使用闪现 -> 惩罚 (应使用)
# 安全时使用闪现 -> 惩罚 (浪费)
action_mask_loss = danger * (1 - flash_probs) + safe * flash_probs
```

**预期效果**: 1小时内解决无效行为，实现闪现的精准释放

---

### 3.2 课程学习 (Curriculum Learning)
**参考**: readme 5.1节、7.1节

**核心思想**: 不同阶段采用不同策略

| 阶段 | 步数比例 | 策略重点 | 宝箱奖励 | 生存权重 |
|------|----------|----------|----------|----------|
| **前期** | 0-30% | 探索收集 | 15.0 | 0.3 |
| **中期** | 30-70% | 平衡 | 10.0 | 0.5 |
| **后期** | 70-100% | 保守生存 | 5.0 | 0.7 |

**实现方式**:
```python
if is_early_phase:
    treasure_weight = 1.0
elif is_late_phase:
    treasure_weight = 0.3
else:
    treasure_weight = 0.7
```

---

### 3.3 情景奖励 (Situational Reward)
**参考**: readme 6节 - **核心技巧，直接提高2个baseline**

**设计原则**: 将稀疏奖励密集化，变相扩增有效样本

| 情景 | 奖励 | 说明 |
|------|------|------|
| 危险时冒险收集宝箱 | **-3.0** | 惩罚冒险行为 |
| 安全时不探索 | **-0.1** | 轻微惩罚保守 |
| 近距离成功逃脱 (<40格) | **+1.0** | 鼓励精彩操作 |
| 闪现成功逃脱 | **+2.0** | 技能使用奖励 |
| 浪费闪现 (安全时使用) | **-1.0** | 惩罚乱用技能 |
| 通关 (存活1000步) | **+10.0** | 生存奖励 |
| 被怪物捕获 | **-10.0** | 死亡惩罚 |

**关键代码**:
```python
# 情景1: 危险时靠近宝箱是冒险行为
if danger > 0.7 and closer_to_treasure:
    reward -= 3.0

# 情景2: 从危险成功逃脱
if escaped and monster_dist < 40:
    reward += 1.0
```

---

### 3.4 FC + GRU 混合网络
**参考**: readme 5.1节 - **FC:GRU = 3:1**

**问题**: 纯LSTM在样本有限时容易过拟合

**解决方案**:
```
输入 (135D)
  ↓
FC Layer 1: 135 → 192, ReLU
  ↓
PSCN模块: 192 → 96 → 192 (残差连接)
  ↓
FC Layer 2: 192 → 192, ReLU
  ↓
GRU: 192 → 64 (1层，捕捉时序)
  ↓
融合: concat(192, 64) = 256 → 128
  ↓
Actor: 128 → 64 → 16 (动作)
Critic: 128 → 64 → 2 (价值)
```

**PSCN (Power Series Connected Network)**: 类似ResNet但参数更少

---

### 3.5 动态超参数调整
**参考**: readme 7.1节调参建议

| 参数 | 初始值 | 终值 | 变化策略 | 目的 |
|------|--------|------|----------|------|
| **Gamma** | 0.995 | 0.9975 | 渐增 | 逐步看重长期回报 |
| **Lambda** | 0.95 | 0.98 | 渐增 | 减少优势估计方差 |
| **Beta (熵)** | 0.015 | 0.001 | 渐减 | 前期探索，后期利用 |
| **学习率** | 3e-4 | 3e-6 | 指数衰减 | 精细收敛 |

**计算公式**:
```python
current_gamma = gamma_start + (gamma_end - gamma_start) * (step / decay_steps)
```

---

### 3.6 Dual-Clip PPO
**参考**: readme 4节

**标准PPO**:
```
ratio = new_prob / old_prob
loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

**Dual-Clip PPO** (增加第二个clip边界):
```
loss = -min(max(ratio * A, clip(ratio, 1-ε, 1+ε) * A), 3.0 * A)
```

**作用**: 防止ratio过大导致训练不稳定

---

### 3.7 多头价值估计
**设计**: 分别估计生存价值和宝箱收集价值

```python
value = [V_survive, V_treasure]

# 加权优势
advantage = 0.7 * A_survive + 0.3 * A_treasure
```

**优势**: 解耦不同目标的估计，提高学习效率

---

## 四、技术实现

### 4.1 项目结构
```
agent_diy/
├── agent.py                 # Agent主类 (GRU状态管理)
├── algorithm/
│   └── algorithm.py         # PPO算法 (Dual-Clip + 动作掩码)
├── conf/
│   ├── conf.py              # 配置 (动态参数)
│   ├── monitor_builder.py   # 监控指标配置
│   └── train_env_conf.toml  # 环境配置
├── feature/
│   ├── definition.py        # 数据结构、GAE、动态参数
│   └── preprocessor.py      # 特征工程 (课程+情景)
├── model/
│   └── model.py             # FC+GRU混合网络 + PSCN
├── workflow/
│   └── train_workflow.py    # 训练流程
└── README_DIY.md            # 技术文档
```

### 4.2 关键配置参数

```python
# 特征维度
FEATURE_DIM = 135        # 6+12+60+12+25+4+16
ACTION_NUM = 16          # 8移动 + 8闪现
VALUE_NUM = 2            # 生存价值 + 宝箱价值

# PPO超参数
GAMMA = 0.995 → 0.9975   # 渐进
LAMDA = 0.95 → 0.98      # 渐进
CLIP_PARAM = 0.2
VF_COEF = 0.5            # policy:value = 2:1
BETA = 0.015 → 0.001     # 渐进

# 网络结构
FC_HIDDEN_DIM = 192
GRU_HIDDEN_DIM = 64      # 3:1比例
```

### 4.3 训练流程

```
每局游戏:
1. Reset Agent (重置GRU隐藏状态)
2. Load latest model
3. While not done (1000步):
   a. 特征处理: env_obs → feature (135D)
   b. 模型推理: feature → action (采样)
   c. 环境交互: action → next_obs, reward
   d. 奖励塑形: reward → shaped_reward (课程+情景)
   e. 收集样本: (obs, act, reward, value, ...)
4. 样本后处理: GAE计算优势 (动态gamma/lambda)
5. PPO更新: 策略损失 + 价值损失 + 熵损失 + 动作掩码损失
6. 定期保存模型
```

---

## 五、预期效果

### 5.1 基于夺冠经验的预期提升

| 改进项 | 参考依据 | 预期效果 |
|--------|----------|----------|
| 动作掩码损失 | readme 4.1节 | 1小时内解决"站着不动"，闪现使用精准化 |
| 情景奖励 | readme 6节 | **直接提升2个baseline**，合作/对抗水平提高 |
| FC+GRU混合 | readme 5.1节 | 防止过拟合，泛化能力提升 |
| 课程学习 | readme 7.1节 | 后期生存率显著提高 |
| 动态超参 | readme 7.1节 | 训练更稳定，收敛更好 |

### 5.2 相比基线版本的优势

| 维度 | 基线 (ppo) | DIY方案 |
|------|------------|---------|
| 动作空间 | 8维 (仅移动) | 16维 (+闪现) |
| 特征维度 | 40维 | 135维 (+宝箱/Buff/地图) |
| 网络结构 | 简单MLP | FC+GRU混合+PSCN |
| 价值估计 | 单头 | 多头 (生存+宝箱) |
| 奖励设计 | 仅生存 | 课程学习+情景奖励 |
| 超参数 | 固定 | 动态调整 |

---

## 六、当前状态

### 6.1 完成情况
- [x] 完整特征工程 (135维)
- [x] FC+GRU混合网络
- [x] Dual-Clip PPO算法
- [x] 动作掩码损失
- [x] 课程学习机制
- [x] 情景奖励设计
- [x] 动态超参数
- [x] 多头价值估计

### 6.2 代码验证
```
✓ Config 加载正常
✓ 特征维度计算正确 (135D)
✓ 模型前向传播正常
✓ 动作采样正常
✓ GRU状态传递正常
✓ 奖励塑形逻辑正常
```

### 6.3 待解决问题
- 开悟平台启动时遇到代理连接问题 (网络配置，非代码问题)
- 需要在实际环境中进行训练和调优

---

## 七、后续优化方向

### 7.1 短期优化
1. **Self-Play**: 使用历史模型作为对手 (readme 4.2节)
2. **模型融合**: Module Soup技术，增强鲁棒性 (readme 4.4节)
3. **注意力机制**: 动态关注重要宝箱

### 7.2 长期扩展
1. **多智能体**: 扩展到3v3场景
2. **分层策略**: 高层决策+低层控制
3. **模仿学习**: 结合专家演示

---

## 八、参考资料

1. **腾讯开悟平台官方文档**
2. **第二届开悟比赛冠军队伍技术分享** (`readme.md`)
   - 动作约束函数思想
   - FC+GRU混合架构
   - 情景奖励设计
   - 调参经验
3. **PPO原始论文**: Proximal Policy Optimization Algorithms (Schulman et al., 2017)
4. **GAE论文**: High-Dimensional Continuous Control Using Generalized Advantage Estimation

---

## 附录：关键代码片段

### A. 情景奖励核心代码
```python
def calculate_situational_rewards(cur_info, next_info):
    reward = np.zeros(2, dtype=np.float32)
    
    # 情景1: 危险时冒险收集 -> 惩罚
    if cur_info['danger_level'] > 0.7 and closer_to_treasure:
        reward[1] -= 3.0
    
    # 情景2: 成功逃脱 -> 奖励
    if next_info['escaped'] and monster_dist < 40:
        reward[0] += 1.0
    
    return reward
```

### B. 动作掩码损失
```python
# 危险时不使用闪现 -> 惩罚
danger_mask = (danger > 0.7).float()
should_use_flash = danger_mask * (1 - flash_used)
loss += (should_use_flash * (1 - flash_probs)).pow(2).mean()

# 安全时使用闪现 -> 惩罚 (浪费)
safe_mask = (danger < 0.2).float()
wasted_flash = safe_mask * flash_used
loss += (wasted_flash * flash_probs).pow(2).mean() * 0.5
```

### C. FC+GRU混合网络
```python
# FC骨干 (3层，占主要计算)
fc_features = self.fc_backbone(obs)  # [batch, 192]

# GRU (1层，64维，防过拟合)
gru_output, _ = self.gru(fc_features.unsqueeze(1), hidden)
gru_features = gru_output.squeeze(1)  # [batch, 64]

# 融合 (3:1比例)
fused = torch.cat([fc_features, gru_features], dim=-1)
```

---

**文档结束**
