## 2025.7.24.

将本次的obs和extra_info保存在`agent_ppo/debug/`文件夹下

## 2025.7.25. v0.1
1. 弃用`agent_ppo/feature/preprocessor.py`，将全部状态和奖励的计算放在`agent_ppo/feature/state_manager.py`中, 全部配置参数写在`agent_ppo/conf/conf.py`
2. 重新设计状态特征:
    1. 英雄当前位置 (2,)
    2. 终点位置 (6,)
    3. SEQUENCE_LENGTH前的英雄位置 (6,)
    4. buff位置 (6,)
    5. treasure位置 (6,) * 13
    6. 闪现是否可用 (1,)
    7. buff是否存在 (1,)
    8. 可用动作 (16,)
    9. 地图障碍物信息 (11*11,)
3. 动作mask维度错误（原来只有8维，更新到16维），包含闪现
4. 设计reward函数
    1. 到终点, 惩罚没有得到的宝箱
    2. 闪现距离奖励 (官方写闪现距离为16个单位)
    3. 撞墙惩罚
    4. 宝箱奖励
    5. buff奖励
    6. 步数惩罚
5. 设计网络架构, 用resnet+layernorm

## 2025.7.26.
### v0.2
测试奖励函数是否存在问题

v0.1训练7.5h仅能找到附近的几个宝箱，且基本都超时到终点，测试奖励函数
1. 修复reward最后一帧奖励重复加入的问题，在日志中加入episode总reward记录
2. 仅保留终点奖励，闪现距离奖励，撞墙惩罚，步数惩罚
3. 学习率改为2.5e-4, 熵系数1e-2
4. 修复推理中存在的exploit函数, 删除extra_info的使用
5. 将diy_5设置为对td return的估计

## 2025.7.27.
### v0.3
1. 奖励中加入对终点移动距离的连续奖励
2. 加入最近宝箱移动距离的连续奖励(未开启)
3. 增大宝箱奖励系数 1.0 -> 10, 有效闪现距离奖励 0.1 -> 0.5, 移动距离奖励 0.1, 撞墙惩罚 0.05 -> 0.1

### v0.4
1. HIT_THRESHOLD写的完全错误, 撞墙就是x, z完全相等 (移动一步距离正好就是1, 加速buff移动一步距为1.5)
2. value loss的数值过大再10到200变化，训练不稳定，需调整连续奖励设计, REW_DISTANCE = 0.1 -> 0.05
3. vf_coef=0.5
4. 修复严重错误, OrganManager如果found过了就不会再更新相对位置, 则奖励计算完全错误
5. 加入新特征 available, 宝箱是否还可以获取

## 2025.7.28.
### v0.5
1. 修复移动就会增加奖励的问题
2. 对闪现的奖励进行clip(-1, 1)

### v0.6
1. 多加一维表示闪现cd
2. 加入宝箱获得与错失惩罚, 数值为1.0
3. 删除超时惩罚, 减小到达终点奖励15 -> 5
4. 删去向目标移动奖励, 删除闪现奖励
5. 减小向目标 (宝箱或终点) 移动距离奖励`REW_DISTANCE: 0.02 -> 0.001`
6. 修正宝箱存在的判断条件

## 2025.7.29.
### v0.7
V0.6仍然无法捡到宝箱可能是没有引导，在没有获得所有宝箱时候，先只给最近的宝箱引导，获得所有宝箱后给终点引导
1. 加入顺序引导，先对最近宝箱进行引导，再对终点进行引导
2. 加入FLASH移动距离奖励和BUFF加速奖励
3. 加入宝箱数依据均匀分布采样

### v0.7.1
v0.7仍然无法捡到更多宝箱，调试一下奖励设计是否合理，是否和状态输入有关
1. 固定8个宝箱，固定宝箱在状态输入的位置
2. 固定起点和终点位置，关闭障碍物生成
> 仅需对[`train_workflow.py`](./code/agent_ppo/workflow/train_workflow.py)中的`run_episodes`函数内的`usr_conf`进行修改


## 2025.7.30.
### v0.7.2
obs加入地图构建 (4, 128, 128):
1. dim 0: 障碍物 (-1, 0, 1) 分别为 (未知, 无, 有)
2. dim 1: 走过的次数 (0 ~ 1) 分别为次数 (0,1,2,...), 作用 clip(x/10,0,1)
3. dim 2: buff/宝箱 (-1, -0.5, 0, 0.5, 1) 分别为 (buff精确位置, buff大致位置, 无, 宝箱大致位置, 宝箱精确位置)
1. dim 3: 终点 (0, 0.5, 1)  分别为 (无, 终点大致位置, 终点精确位置)
2. dim 4: 英雄位置 (0, 1) 分别为 (无, 有)
5. 网络使用ResNet处理图像信息, 网络大小增大到3Mb -> 20Mb
6. obs简化为:
    1. 英雄当前位置 (2,)
    2. 闪现是否可用 (1,)
    3. 闪现cd (1,)
    4. buff是否存在 (1,)
    5. 地图信息 (4, 128, 128)
7. 修复英雄位置忘记重置的问题
8. 加入周围重复步数惩罚, 周围5x5的移动次数, 当超过阈值`REW_MEMORY_PUNISH_STEP=2`时则按照比例矩阵进行惩罚, 并将惩罚clip(0, 0.2)
    ```python
    [0.1, 0.2, 0.5, 0.2, 0.1],
    [0.2, 0.5, 0.8, 0.5, 0.2],
    [0.5, 0.8, 1.0, 0.8, 0.5],
    [0.2, 0.5, 0.8, 0.5, 0.2],
    [0.1, 0.2, 0.5, 0.2, 0.1],
    ```
9. 加入全局奖励缩放系数: `REW_GLOBAL_SCALE=1.0`
10. 发现未发现状态下`REW_DISTANCE`一次变化的距离可能非常大，因为离散的delta距离为20，因此将离散的距离都做clip(-1, 1)

## 2025.7.31.
### v0.7.3
v0.7.2的全局地图特征可能并不好找到离散的英雄位置和宝箱位置，重新用全链接网络，obs沿用v0.7.1
1. 将网络重新用回离散宝箱位置，宝箱按照从近到远距离进行排序作为状态输入
2. 保留周围重复走路惩罚
3. 将周围重复路次数按照17x17加入状态中
4. 删去位置特征中的available标签, 当宝箱或buff不avail时, 则直接返回全0, 相对位置1的特征

## 2025.8.1.
### v0.7.4
v0.7.3奖励设置还是有问题，先是到终点很高胜率，后期就开始下降，但总奖励没有下降很多
1. 使用去年的奖励设计，仅有到达终点后才计算宝箱惩罚，超时不用宝箱惩罚，最后全部奖励乘上系数0.1
2. buff获得奖励按照次数`*0.5**cnt`递减
3. 熵系数用1e-4

### v0.8
发现严重问题
1. 一个organ找到后如果又变为未找到，则位置就会忘记，但是又按照估计的距离进行处理，导致之前记住的位置又变为估计的，导致距离奖励计算错误
2. 宝箱拾取也是一个道理，捡到一次后status=0但是没见过后他的status又变为-1 (buff也就按照宝箱捡一次处理了)
3. 熵系数用回1e-2
> 如果要本地加载模型, 就在`agent._predict`中加载一次即可

### v0.8.1
1. 将所有奖励*0.1，可以不用系数
2. `REW_MEMORY_PUNISH_STEP = 4 -> 2`
3. 网络为纯MLP
4. 奖励一直包含向目标点距离的奖励
5. 增大周围重复步数惩罚

## 2025.8.2.
### v0.8.2
1. 减小batch size, buffer size, 训练频率, 模型保存速度

### v0.8.3
1. 修复地图计算错误, 重新用CNN网络处理地图特征

### v0.9
1. 将模型换为target_dqn完成不撞墙测试，网络和特征可用
2. 完成宝箱,buff,end位置压缩到视野内的特征

## 2025.8.3 (剩余6天)
### v0.9.1
v0.9已经能稳定到达终点, 但宝箱总是少一个收集, 需要改奖励
(不能删除) 删除到达终点奖励, 增大获得宝箱奖励`10->20`
1. 先引导到宝箱, 再到终点
2. 网络结构中分别加入一个mlp和cnn的resnet模块
3. DDQN的`EPSILON_DECAY: 1e-6 -> 1e-4`

加入具身赛道的target_dqn代码，修改部分:
1. 可能都没发现全部宝箱就走到终点, 因此惩罚宝箱需要通过总数做减法得到

### v0.9.2 (放弃)
- 王者赛道
1. 在王者环境上测试使用记忆化地图, 即将obstacles地图和模型一起保存下来, 加载也和模型一同加载, 并每次在之前的地图基础上增加
2. 模型输入为整个地图 (先在简单避障任务上进行测试), actor用cpu预测速度太慢, 导致环境崩溃


### v1.0
- 两个赛道
1. 放弃cnn中的resnet模块
2. DDQN中的`EPSILON_DECY`是按照预测次数进行递降的, 因此设置为`1e-6`合理
3. 沿用宝箱和终点奖励, 否则到不了终点, 胜率很低

- 具身赛道
1. 加入探索未知区域奖励, 每一步对的多未知区域探索基于奖励

## 2025.8.4 (剩余5天)
### v1.1
- 两个赛道: 都不走到终点, 修改奖励如下
1. (未启用, 中断还是`-REW_FINISH`) 加入中断惩罚`-80`即所有获得过的宝箱
2. 状态中加入当前`步数/1000`
3. 修改周围步数惩罚计算方法, 统计周围3x3范围的总步数, 减去阈值9, 乘上惩罚系数0.1得到周围步数惩罚 (并clip到1.0)
4. 保留同时对终点和宝箱的实时距离奖励

- 测试条件:
1. 测试都是仅包含最简的避免撞墙和不走重复路奖励
2. 环境数为4
3. 环境参数为固定位置fixed
4. 王者赛道
- 测试目标: 到达稳定不撞墙的状态第一时间 (wty 笔记本)
- 测试内容:
1. 测试2048和512的batchsize有没有很大的区别, 2048 batchsize: 12mins, 512 batchsize: 16mins
2. 测试将整个地图作为obs输入只能支持512 batchsize并且效果很差, 基本训练不出来
3. 测试带有resnet和不带的效果, 12min不带resnet收敛, 30mins带resnet几乎收敛, 于是将网络退回不带resnet结构的网络

- 测试条件:
1. 随机环境下, 王者赛道, v1.1版本
2. epsilon=0.1
- 测试内容
1. 仅用到终点, 不撞墙, 步数, 重复步数惩罚, 终点距离奖励, 测试稳定到终点所需时间
2. TODO: 测试不同的buff size, 10000和100000

### v1.2
1. 加入中断惩罚`-80`即所有宝箱
2. replay buff增大到`1e4 -> 1e5`
3. 错失宝箱惩罚乘上当前胜率系数

### v1.3 (only rob_prelim)
1. 加入simbaV2和ddqn实现(未实装，需要自行修改algorithm_simba_ddqn.py和model_simba.py为algorithm.py和model.py)
2. max_step调整为2000

### v1.3 (only hok_prelim)
1. 加入simbaV2和ddqn实现
2. max_step调味2000
3. gamma: 0.9 -> 0.995
4. 删除miss宝箱的胜率系数 (可能导致宝箱没完全获取)

## 去年的一些设计
### 去年的状态设计
```python
def obs2dict(raw_obs):
  obs = {
    # in (0, 1) = absolute coord / 64000, (64000 = 128 * 500), float
    'norm_pos': np.array((raw_obs.feature.norm_pos.x, raw_obs.feature.norm_pos.z), np.float32),
    # in (128, 128), used for drawing, int
    'grid_pos': np.array((raw_obs.feature.grid_pos.x, raw_obs.feature.grid_pos.z), np.int32),
    ### Relative Position ###
    'start_pos': relative2dict(raw_obs.feature.start_pos),
    'end_pos': relative2dict(raw_obs.feature.end_pos),
    'buff_pos': relative2dict(raw_obs.feature.buff_pos),
    'treasure_pos': [relative2dict(pos) for pos in raw_obs.feature.treasure_pos],
    ### Map Figure 51x51 ###
    'obstacle_map': np.array(raw_obs.feature.obstacle_map, np.int32).reshape(51, 51),
    'memory_map': np.array(raw_obs.feature.memory_map, np.float64).reshape(51, 51),
    'treasure_map': np.array(raw_obs.feature.treasure_map, np.int32).reshape(51, 51),
    'end_map': np.array(raw_obs.feature.end_map, np.int32).reshape(51, 51),
    # Mask for available action, shape=(2,) always legal_act[0]=1, if legal_act[1]=1 skill is available
    'legal_act': np.array(raw_obs.legal_act, np.int32),
  }
  treasure_flags, treasure_grid_distance = np.zeros(13, bool), np.zeros(13, np.float32)
  for i, treasure in enumerate(obs['treasure_pos'][2:]):
    treasure_flags[i] = treasure['direction'] != 0
    treasure_grid_distance[i] = treasure['grid_distance']
  buff_flag = obs['buff_pos']['direction'] != 0
  obs.update({
     'treasure_flags': treasure_flags,
     'treasure_grid_distance': treasure_grid_distance,
     'buff_flag': buff_flag
  })
  return obs

def observation_process(raw_obs, env_info=None):
  """
  env_info: useless, but higher than v9.2.2 this variable must be given.
  """
  obs = obs2dict(raw_obs)
  feature = np.hstack([
    np.stack([  # Image: (4, 51, 51)
      obs['obstacle_map'],
      obs['memory_map'],
      obs['treasure_map'],
      obs['end_map'],
    ], axis=0).reshape(-1),
    *obs['norm_pos'],
    *obs['treasure_flags'],
    *obs['treasure_grid_distance'],
    obs['buff_flag'],
    obs['buff_pos']['grid_distance'],
    obs['end_pos']['grid_distance'],
    obs['legal_act'][1],  # flash available
  ]).astype(np.float32)
  assert feature.shape[0] == args.obs_dim, f"ERROR: {feature.shape[0]=} != {args.obs_dim+1}"
  return ObsData(feature=feature)
```

### 去年的奖励设计
```python
### Reward ###
r = 0
# 1. punish repeated step around (sum(-weight*max(0, repeat_time-1) * 0.1))
ratio = self.total_timestep * args.num_envs / args.total_timesteps
if ratio < 0.5:
    r -= max(obs['memory_map'][25,25] - args.repeat_step_thre, 0)
    # assert obs['memory_map'][25,25] > 0  # won't be > 0
else:
    r -= (args.repeat_punish * np.maximum(
    obs['memory_map'][23:28,23:28]-args.repeat_step_thre, 0)).sum()
# 2. go to end
if terminated:
    r += 150
    # punish treasures haven't get
    r -= (obs['treasure_flags'] * self.treasure_reward_coef).sum() * 100
    # punish buff haven't get
    r -= obs['buff_flag'] * args.forget_buff_punish
    self.treasure_miss_cnt += obs['treasure_flags'].astype(np.int32)
dist_reward_coef = args.flash_dist_reward_coef if self.use_flash else args.dist_reward_coef
delta_end_distance = self._obs['end_pos']['grid_distance'] - obs['end_pos']['grid_distance']
r += delta_end_distance * dist_reward_coef
# 3. treasure
if not terminated and score == 100:
    r += 100 * (self.treasure_reward_coef * (self._obs['treasure_flags'] ^ obs['treasure_flags'])).sum()
if sum(self._obs['treasure_flags']):
    dist_treasure = np.max((self._obs['treasure_grid_distance'] -
                            self.obs['treasure_grid_distance']
                        )[self._obs['treasure_flags']])
    r += dist_treasure * dist_reward_coef
# 4. hit wall
if hit_wall:
    r -= args.flash_hit_wall_punish if self.use_flash else args.walk_hit_wall_punish
# 5. buff
if self._obs['buff_flag'] - obs['buff_flag'] == 1:
    r += args.get_buff_reward
# 6. add each step punish
if ratio > 0.5 or args.load_model_id is not None:
    r -= args.each_step_punish
# 7. add global coef
r *= args.reward_global_coef
```

### 去年的参数
```python
class Args:
    version = "1.2.2_dppo_pretrain_1.2.1_955"  # model version

    # Algorithm specific arguments
    total_timesteps = int(1e7)  # total timesteps of the experiments
    # total_timesteps = int(8000*2)  # total timesteps of the experiments
    learning_rate = 1e-5  # the learning rate of the optimizer 这个可以在稳定之后下降，比如除以十
    # group_lr = [2.5e-4, 2.5e-5]  # adjust learning rate by the proportion 按照步数比例调整学习率(在不启动退火的前提下)
    num_envs = 10  # the number of parallel environments
    # num_envs = 4  # the number of parallel environments
    num_steps = 256  # the number of steps to run in each environment per policy rollout 这个最好大于一个episode的长度，设置成512或者1024
    # num_steps = 32  # the number of steps to run in each environment per policy rollout 这个最好大于一个episode的长度，设置成512或者1024
    anneal_lr = False  # whether to anneal the learning rate or not 是否退火
    gamma = 0.999  # the discount factor gamma 这个要再高一些 0.999左右，也可以是1
    gae_lambda = 0.95  # the lambda for the general advantage estimation
    minibatch_size = 64  # the mini-batch size in batch size (n_sample*n_steps)
    update_epochs = 2  # the K epochs to update the policy
    norm_adv = False  # advantages normalization 可能不需要
    clip_coef = 0.2  # the surrogate clipping coefficient
    clip_vloss = False  # whether or not to use a clipped loss for the value function, as per the paper
    ent_coef = 0  # entropy coefficient 一般设置成1e-4，在训练最优模型的时候设置成0
    # group_ent_coef = [1e-2, 1e-4]  # adjust entropy coefficient by proportion
    vf_coef = 0.5  # value function coefficient
    max_grad_norm = 0.5  # the maximum norm for the gradient clipping
    target_kl = None  # the target KL divergence threshold

    # to be filled in runtime
    batch_size = 0  # the batch size (computed in runtime) 
    num_iterations = 0  # the number of iterations (computed in runtime)

    ### Network ###
    observation_img_shape = (4, 51, 51)
    observation_vec_shape = (31,)
    # image + vec + is_flash_available (mask)
    obs_dim = np.prod(observation_img_shape) + observation_vec_shape[0] + 1
    ### Environment ###
    # n_treasure = ['norm', 13, 'norm']
    n_treasure = "uniform"
    norm_sigma = 3.0
    ### Reward ###
    # distance
    dist_reward_coef = 1.0
    flash_dist_reward_coef = 5.0
    # repeat walk
    repeat_punish = np.array([
        [0, 0, 0, 0, 0],
        [0, 0.5, 0.8, 0.5, 0],
        [0, 0.8, 1.0, 0.8, 0],
        [0, 0.5, 0.8, 0.5, 0],
        [0, 0, 0, 0, 0],
    ], np.float32),
    # treasure
    treasure_miss_reset_episode = 100
    # repeat step
    repeat_step_thre = 0.4
    # hit wall
    walk_hit_wall_punish = 1.0
    flash_hit_wall_punish = 10.0
    # buff
    get_buff_reward = 5.0
    forget_buff_punish = 5.0
    # global coefficient
    reward_global_coef = 1 / 10
    # step punish (only consider when ratio > 0.5)
    each_step_punish = 0.2
    # dynamic treasure reward
    dynamic_treasure_reward = False
    # random start position
    random_start_position_ratio = 0.0
    ### Load Model ###
    load_model_id = "955"
```
