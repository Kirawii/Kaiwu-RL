#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
)

import random
import numpy as np
from pathlib import Path
from kaiwu_agent.utils.common_func import attached
from agent_ppo.model.model import NetworkModelActor
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.definition import SampleData, ObsData, ActData, SampleManager
from agent_ppo.feature.state_manager import StateManager
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger
from kaiwudrl.common.monitor.monitor_proxy_process import MonitorProxy


def random_choice(p):
    r = random.random() * sum(p)
    s = 0
    for i in range(len(p)):
        if r > s and r <= s + p[i]:
            return i, p[i]
        s += p[i]
    return len(p) - 1, p[len(p) - 1]


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger: KaiwuLogger=None, monitor: MonitorProxy=None):
        super().__init__(agent_type, device, logger, monitor)

        self.model = NetworkModelActor()
        self.algorithm = Algorithm(device=device, logger=logger, monitor=monitor)
        self.state_manager = StateManager()
        self.sample_manager = SampleManager()
        self.win_history = []
        self.logger = logger
        self.reset()

    def _predict(self, obs, legal_action):
        # if self.last_action == -1:  # 手动加载一次模型, 稳定加载进去
        #     self.model.load_state_dict(torch.load("/data/projects/back_to_the_realm_v2/ckpt/model.ckpt-47292.pkl", map_location="cpu"))
        #     self.logger.info(f"load model in _predict successfully!!!")
        with torch.no_grad():
            inputs = self.model.format_data(obs, legal_action)
            output_list = self.model(*inputs)

        np_output_list = []
        for output in output_list:
            np_output_list.append(output.numpy().flatten())

        return np_output_list

    def predict_process(self, obs, legal_action):
        obs = np.array([obs])
        legal_action = np.array([legal_action])
        probs, value = self._predict(obs, legal_action)
        return probs, value

    def observation_process(self, obs, terminated=False, truncated=False, extra_info=None):
        self.state_manager.update([obs, extra_info], self.last_action, terminated, truncated)
        # FOR DEBUG
        # self.state_manager.save_frame()
        feature, legal_action, reward = self.state_manager.get_all()

        return ObsData(
            feature=feature,
            legal_action=legal_action,
            reward=reward,
        )

    @predict_wrapper
    def predict(self, list_obs_data):
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action
        probs, value = self.predict_process(feature, legal_action)
        action, prob = random_choice(probs)
        self.last_action = action
        return [ActData(probs=probs, value=value, action=action, prob=prob)]

    def action_process(self, act_data):
        self.last_action = act_data.action
        return act_data.action

    @exploit_wrapper
    def exploit(self, observation):
        # 在平台上验证的时候需要用到该函数
        obs_data = self.observation_process(observation["obs"], extra_info=observation["extra_info"])
        feature = obs_data.feature
        legal_action = obs_data.legal_action
        probs, value = self.predict_process(feature, legal_action)
        action, prob = random_choice(probs)
        act = self.action_process(ActData(probs=probs, value=value, action=action, prob=prob))
        return act

    def reset(self):
        self.state_manager.reset()
        self.last_prob = 0
        self.last_action = -1

    @learn_wrapper
    def learn(self, list_sample_data):
        self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.algorithm.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if not Path(model_file_path).exists():
            self.logger.warning(f"model file {model_file_path} not exists, skip loading")
            return
        self.model.load_state_dict(torch.load(model_file_path, map_location="cpu"))
        self.logger.info(f"load model {model_file_path} successfully")
