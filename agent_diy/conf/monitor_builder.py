#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

监控指标构建器 - 定义DIY Agent的监控指标
包含多头奖励、宝箱收集等指标
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    创建自定义指标的监控面板配置
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("峡谷追猎 - DIY Agent")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        # 生存奖励
        .add_panel(
            name="生存奖励",
            name_en="survive_reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward_survive",
            expr="avg(reward_survive{})",
        )
        .end_panel()
        # 宝箱奖励
        .add_panel(
            name="宝箱奖励",
            name_en="treasure_reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward_treasure",
            expr="avg(reward_treasure{})",
        )
        .end_panel()
        # 总损失
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        # 价值损失
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        # 策略损失
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        # 熵
        .add_panel(
            name="策略熵",
            name_en="entropy",
            type="line",
        )
        .add_metric(
            metrics_name="entropy",
            expr="avg(entropy{})",
        )
        .end_panel()
        .end_group()

        # 环境指标组
        .add_group(
            group_name="环境指标",
            group_name_en="environment",
        )
        # 总得分
        .add_panel(
            name="总得分",
            name_en="total_score",
            type="line",
        )
        .add_metric(
            metrics_name="total_score",
            expr="avg(total_score{})",
        )
        .end_panel()
        # 宝箱收集数
        .add_panel(
            name="宝箱收集数",
            name_en="treasures_collected",
            type="line",
        )
        .add_metric(
            metrics_name="treasures_collected",
            expr="avg(treasures_collected{})",
        )
        .end_panel()
        # 对局步数
        .add_panel(
            name="对局步数",
            name_en="episode_steps",
            type="line",
        )
        .add_metric(
            metrics_name="episode_steps",
            expr="avg(episode_steps{})",
        )
        .end_panel()
        # 闪现使用次数
        .add_panel(
            name="闪现使用",
            name_en="flash_count",
            type="line",
        )
        .add_metric(
            metrics_name="flash_count",
            expr="avg(flash_count{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
