#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

峡谷追猎 - DIY Agent PPO 算法实现
优化版本：动作掩码损失 + 动态熵系数 + Dual-Clip PPO
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import get_current_beta


class Algorithm:
    """
    PPO (Proximal Policy Optimization) 算法实现 - 优化版本

    改进点:
    1. 动作掩码损失 (Action Mask Loss) - 惩罚浪费闪现等行为
    2. 动态熵系数 (Dynamic Entropy) - 逐步衰减
    3. Dual-Clip PPO - 更稳定的训练
    4. 辅助任务损失 (危险预测)

    损失组成:
        total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss
                   + action_mask_loss + auxiliary_loss
    """

    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        # 配置参数
        self.action_num = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM

        # 训练步数统计
        self.train_step = 0
        self.last_report_monitor_time = 0

        # 是否使用 dual-clip
        self.use_dual_clip = True
        self.dual_clip_coef = 3.0  # 第二个clip边界

    def learn(self, list_sample_data):
        """
        训练入口: 对一批样本执行PPO更新

        Args:
            list_sample_data: SampleData 列表
        """
        # 准备数据
        obs = torch.stack([torch.tensor(f.obs, dtype=torch.float32) for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([torch.tensor(f.legal_action, dtype=torch.float32) for f in list_sample_data]).to(self.device)
        act = torch.stack([torch.tensor(f.act, dtype=torch.long) for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([torch.tensor(f.prob, dtype=torch.float32) for f in list_sample_data]).to(self.device)
        advantage = torch.stack([torch.tensor(f.advantage, dtype=torch.float32) for f in list_sample_data]).to(self.device)
        old_value = torch.stack([torch.tensor(f.value, dtype=torch.float32) for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([torch.tensor(f.reward_sum, dtype=torch.float32) for f in list_sample_data]).to(self.device)

        # 准备 map_tensor (如果样本中有)
        map_tensors = []
        for f in list_sample_data:
            if hasattr(f, 'map_tensor') and f.map_tensor is not None:
                map_tensors.append(torch.tensor(f.map_tensor, dtype=torch.float32))
            else:
                map_tensors.append(torch.zeros((4, 51, 51), dtype=torch.float32))
        map_tensor = torch.stack(map_tensors).to(self.device) if map_tensors else None

        # 获取危险等级 (用于动作掩码损失)
        danger_levels = self._extract_danger_levels(list_sample_data)
        flash_used = self._extract_flash_usage(list_sample_data)

        # 更新动态熵系数
        self.var_beta = get_current_beta(self.train_step)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        # 前向传播 (带辅助任务和map_tensor)
        hidden_state = self.model.get_initial_hidden(batch_size=obs.size(0), device=self.device)
        logits, value_pred, _, danger_pred, _, _ = self.model(obs, map_tensor=map_tensor, inference=False, hidden_state=hidden_state)

        # 计算损失
        total_loss, info_dict = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            danger_levels=danger_levels,
            flash_used=flash_used,
            danger_pred=danger_pred,
        )

        # 反向传播和优化
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()

        self.train_step += 1

        # 定期上报监控指标
        self._report_monitor(total_loss, info_dict, list_sample_data)

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        danger_levels,
        flash_used,
        danger_pred,
    ):
        """
        计算PPO损失 (策略损失 + 价值损失 + 熵正则 + 动作掩码损失)

        Returns:
            total_loss: 总损失
            info_dict: 各损失分量的字典
        """
        batch_size = logits.size(0)

        # ==================== 动作掩码 Softmax ====================
        prob_dist = self._masked_softmax(logits, legal_action)

        # ==================== 策略损失 (Dual-Clip PPO) ====================
        # 计算新策略的动作概率
        one_hot = F.one_hot(old_action.squeeze(-1), self.action_num).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True).clamp(1e-9)

        # 旧策略概率
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)

        # 重要性采样比率
        ratio = new_prob / old_action_prob

        # 优势函数 (1维)
        adv = advantage.view(-1, 1)

        # Dual-Clip PPO
        surr1 = ratio * adv
        surr2 = ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv

        # 第二个clip (防止ratio过大)
        if self.use_dual_clip:
            surr3 = self.dual_clip_coef * adv  # 3倍上限
            policy_loss = -torch.min(
                torch.max(surr1, surr2),
                surr3
            ).mean()
        else:
            policy_loss = -torch.min(surr1, surr2).mean()

        # ==================== 价值损失 (Clipped) ====================
        # value_pred是 [batch, 1]，old_value和reward_sum也是 [batch, 1] 或 [batch]
        vp = value_pred.view(-1)
        ov = old_value.view(-1)
        tdret = reward_sum.view(-1)

        value_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss1 = F.mse_loss(vp, tdret, reduction='none')
        value_loss2 = F.mse_loss(value_clip, tdret, reduction='none')
        value_loss = torch.max(value_loss1, value_loss2).mean()

        # ==================== 熵损失 ====================
        entropy = (-prob_dist * torch.log(prob_dist.clamp(1e-9))).sum(1).mean()
        entropy_loss = -entropy

        # ==================== 动作掩码损失 (核心改进) ====================
        # 参考夺冠经验：惩罚危险时不用闪现，或安全时用闪现
        action_mask_loss = torch.tensor(0.0, device=logits.device)

        if danger_levels is not None and flash_used is not None:
            # 提取闪现动作的概率 (8-15)
            flash_probs = prob_dist[:, 8:].sum(dim=1)  # [batch]

            # 危险等级转换为tensor
            danger_t = torch.tensor(danger_levels, dtype=torch.float32, device=logits.device)
            flash_t = torch.tensor(flash_used, dtype=torch.float32, device=logits.device)

            # 情景1: 危险时不使用闪现 -> 鼓励使用
            # loss = (danger_high * (1 - flash_used)) ^ 2
            danger_mask = (danger_t > Config.DANGER_THRESHOLD_HIGH).float()
            should_use_flash = danger_mask * (1 - flash_t)
            action_mask_loss += (should_use_flash * (1 - flash_probs)).pow(2).mean() * Config.ACTION_MASK_LOSS_COEF

            # 情景2: 安全时使用闪现 -> 惩罚浪费
            # loss = (danger_low * flash_used) ^ 2
            safe_mask = (danger_t < Config.DANGER_THRESHOLD_LOW).float()
            wasted_flash = safe_mask * flash_t
            action_mask_loss += (wasted_flash * flash_probs).pow(2).mean() * Config.FLASH_WASTE_PENALTY

        # ==================== 辅助任务损失 (危险预测) ====================
        aux_loss = torch.tensor(0.0, device=logits.device)
        if danger_levels is not None and danger_pred is not None:
            danger_target = torch.tensor(danger_levels, dtype=torch.float32, device=logits.device).view(-1, 1)
            aux_loss = F.mse_loss(danger_pred, danger_target)

        # ==================== 总损失 ====================
        total_loss = (
            self.vf_coef * value_loss +
            policy_loss +
            self.var_beta * entropy_loss +
            action_mask_loss +
            0.1 * aux_loss  # 辅助任务权重较小
        )

        info_dict = {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy': entropy.item(),
            'action_mask_loss': action_mask_loss.item(),
            'aux_loss': aux_loss.item(),
            'beta': self.var_beta,
        }

        return total_loss, info_dict

    def _masked_softmax(self, logits, legal_action):
        """合法动作掩码下的 softmax"""
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return F.softmax(label, dim=1)

    def _extract_danger_levels(self, list_sample_data):
        """从样本中提取危险等级 (简化版本，实际可以从obs中解析)"""
        # 这里简化处理，实际可以从特征中解析
        # 特征的最后几位包含进度信息
        danger_levels = []
        for sample in list_sample_data:
            # 从特征中解析危险等级 (假设在固定位置)
            # 实际实现中应该在sample中存储danger_level
            danger_levels.append(0.5)  # 默认值
        return danger_levels

    def _extract_flash_usage(self, list_sample_data):
        """从样本中提取是否使用闪现"""
        flash_used = []
        for sample in list_sample_data:
            action = sample.act if hasattr(sample, 'act') else 0
            flash_used.append(1.0 if action >= 8 else 0.0)
        return flash_used

    def _report_monitor(self, total_loss, info_dict, list_sample_data):
        """上报监控指标"""
        now = time.time()
        if now - self.last_report_monitor_time < 60:
            return

        # 计算平均奖励
        rewards = np.array([s.reward for s in list_sample_data])
        reward_mean = round(rewards.mean(), 4) if len(rewards) > 0 else 0.0

        results = {
            "total_loss": round(total_loss.item(), 4),
            "value_loss": round(info_dict['value_loss'], 4),
            "policy_loss": round(info_dict['policy_loss'], 4),
            "entropy": round(info_dict['entropy'], 4),
            "entropy_loss": round(info_dict['entropy_loss'], 4),
            "action_mask_loss": round(info_dict['action_mask_loss'], 4),
            "aux_loss": round(info_dict['aux_loss'], 4),
            "beta": round(info_dict['beta'], 6),
            "reward": reward_mean,
        }

        self.logger.info(
            f"[train] step:{self.train_step} "
            f"total:{results['total_loss']} "
            f"policy:{results['policy_loss']} "
            f"value:{results['value_loss']} "
            f"entropy:{results['entropy']} "
            f"mask:{results['action_mask_loss']} "
            f"beta:{results['beta']}"
        )

        if self.monitor:
            self.monitor.put_data({os.getpid(): results})

        self.last_report_monitor_time = now
