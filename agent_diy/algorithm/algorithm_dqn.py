#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

峡谷追猎 - DIY Agent DQN 算法实现
完全参考 agent_target_dqn 亚军方案
"""

import time
import os
import numpy as np
import torch
from copy import deepcopy
from agent_diy.conf.conf import Config
from agent_diy.model.model import Model
from agent_diy.model.simbaV2.agents.networks import l2normalize_network


class Algorithm:
    """
    DQN (Deep Q-Network) 算法实现
    使用Target Network和Epsilon-Greedy探索
    """

    def __init__(self, device, logger, monitor):
        self.act_shape = Config.ACTION_NUM
        self.direction_space = Config.MOVE_ACTION_NUM
        self.talent_direction = Config.FLASH_ACTION_NUM
        self.obs_shape = Config.DIM_OF_OBSERVATION
        self.epsilon_max = Config.EPSILON_MAX
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self._gamma = Config.GAMMA
        self.lr = Config.START_LR
        self.device = device

        # 创建模型
        self.model = Model(device=device)
        self.model.to(self.device)

        # 优化器
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Target网络
        self.target_model = deepcopy(self.model)

        # 统计
        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0
        self.logger = logger
        self.monitor = monitor

    def learn(self, list_sample_data):
        """
        训练入口

        Args:
            list_sample_data: SampleData列表
        """
        t_data = list_sample_data
        batch = len(t_data)

        # 准备数据
        batch_feature_vec = [frame.obs for frame in t_data]
        batch_action = torch.LongTensor(np.array([int(frame.act) for frame in t_data])).view(-1, 1).to(self.device)

        # 下一状态的合法动作掩码
        _batch_obs_legal = torch.stack([
            torch.tensor(
                frame._obs_legal if (hasattr(frame, '_obs_legal') and frame._obs_legal is not None)
                else np.ones(Config.ACTION_NUM, dtype=np.float32),
                dtype=torch.bool
            )
            for frame in t_data
        ]).to(self.device)

        batch_feature = self.__convert_to_tensor(batch_feature_vec)

        # 准备下一状态特征
        _batch_feature_vec = []
        for frame in t_data:
            if hasattr(frame, '_obs') and frame._obs is not None:
                _batch_feature_vec.append(frame._obs)
            else:
                # 如果没有next_obs，使用当前obs（终局情况）
                _batch_feature_vec.append(frame.obs)

        _batch_feature = self.__convert_to_tensor(_batch_feature_vec)

        # 奖励和完成标记
        rew = torch.tensor([
            frame.rew[0] if hasattr(frame.rew, '__getitem__') else frame.rew
            for frame in t_data
        ], device=self.device, dtype=torch.float32)

        not_done = torch.tensor([
            0.0 if (frame.done[0] if hasattr(frame.done, '__getitem__') else frame.done) == 1.0 else 1.0
            for frame in t_data
        ], device=self.device, dtype=torch.float32)

        self.optim.zero_grad()

        model = self.model
        model.train()
        target_model = self.target_model
        target_model.eval()

        with torch.no_grad():
            # 使用当前模型选择动作 (Double DQN)
            q = model.get_q_values(_batch_feature)
            q = q.masked_fill(~_batch_obs_legal, float(torch.min(q)))

            # 使用Target网络计算Target Q值
            q_t = target_model.get_q_values(_batch_feature)
            q_max = q_t.gather(dim=-1, index=q.argmax(dim=-1, keepdim=True)).squeeze(-1).detach()

        # 计算当前Q值
        logits = model.get_q_values(batch_feature)
        target_q = rew + self._gamma * q_max * not_done
        loss = torch.square(target_q - logits.gather(1, batch_action).view(-1)).mean()

        loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optim.step()

        # L2归一化 (SimbaV2关键)
        if hasattr(model, 'l2_normalize'):
            model.l2_normalize()

        self.train_step += 1

        # 更新Target网络
        if self.train_step % self.target_update_freq == 0:
            self.update_target_q()

        value_loss = loss.detach().item()
        q_value = target_q.mean().detach().item()
        reward = rew.mean().detach().item()

        # 定期上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": round(value_loss, 4),
                "q_value": round(q_value, 4),
                "reward": round(reward, 4),
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            if self.logger:
                self.logger.info(
                    f"[train] step:{self.train_step} "
                    f"loss:{value_loss:.4f} q:{q_value:.4f} reward:{reward:.4f}"
                )

            self.last_report_monitor_time = now

    def __convert_to_tensor(self, data):
        """将数据转换为tensor"""
        if isinstance(data, list):
            if isinstance(data[0], torch.Tensor):
                processed = torch.stack(data, dim=0).to(self.device).float()
            elif isinstance(data[0], np.ndarray):
                processed = torch.from_numpy(np.stack(data, axis=0)).to(self.device).float()
            else:
                processed = torch.tensor(np.array(data), dtype=torch.float32).to(self.device)
        elif isinstance(data, np.ndarray):
            processed = torch.from_numpy(data.astype(np.float32)).to(self.device)
        elif torch.is_tensor(data):
            processed = data.to(self.device).float()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        return processed

    def predict_detail(self, list_obs_data, exploit_flag=False):
        """
        预测动作

        Args:
            list_obs_data: ObsData列表
            exploit_flag: 是否纯利用模式（无探索）

        Returns:
            动作列表
        """
        batch = len(list_obs_data)

        feature_vec = [obs_data.feature for obs_data in list_obs_data]
        legal_act = [obs_data.legal_act for obs_data in list_obs_data]
        legal_act = torch.tensor(np.array(legal_act)).bool().to(self.device)

        model = self.model
        model.eval()

        # 计算epsilon
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(
            -self.epsilon_decay * self.predict_count
        )

        with torch.no_grad():
            # Epsilon-Greedy
            if not exploit_flag and np.random.rand(1) < self.epsilon:
                # 随机探索 (在合法动作中)
                random_action = np.random.rand(batch, self.act_shape)
                random_action = torch.tensor(random_action, dtype=torch.float32).to(self.device)
                random_action = random_action.masked_fill(~legal_act, 0)
                act = random_action.argmax(dim=1).cpu().view(-1, 1).tolist()
            else:
                # 贪婪选择
                feature = self.__convert_to_tensor(feature_vec)
                logits = model.get_q_values(feature)
                logits = logits.masked_fill(~legal_act, float(torch.min(logits)))
                act = logits.argmax(dim=1).cpu().view(-1, 1).tolist()

        # 转换动作格式: [move_dir, use_talent]
        # move_dir: 0-7 (方向), use_talent: 0或1 (是否使用闪现)
        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]
        self.predict_count += 1

        # 构建返回
        from agent_diy.feature.definition import ActData
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    def update_target_q(self):
        """更新Target网络"""
        self.target_model.load_state_dict(self.model.state_dict())
        if self.logger:
            self.logger.info(f"[train] Target network updated at step {self.train_step}")
