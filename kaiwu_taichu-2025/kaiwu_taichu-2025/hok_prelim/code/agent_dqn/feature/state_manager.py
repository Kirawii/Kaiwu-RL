"""
将环境返回的frame_state (Observation, ExtraInfo['game_info'])转为我们所需的obs特征并计算reward函数
"""

import numpy as np
from collections import deque
from agent_dqn.conf.conf import Config as cfg
from agent_dqn.utils.display_iterable_struct import save_json, simplify_iter, too_simplify_iter
from agent_dqn.utils import PATH_DEBUG

PATH_FRAMES = PATH_DEBUG / "frames"
PATH_FRAMES.mkdir(parents=True, exist_ok=True)

class StateManager:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.n_step = 0
        self.curr_frame = None
        self.prev_frames = deque(maxlen=cfg.sequence_length)
    
    def update(self, frame_state: list):
        """
        Args:
            frame_state: list, 环境返回的帧状态信息, [0]: Observation, [1]: ExtraInfo
        """
        self.n_step += 1
        self.prev_frames.append(self.curr_frame)
        self.curr_frame = {
            'obs': frame_state[0],
            'extra_info': frame_state[1]
        }
    
    def get_reward(self) -> float:
        ...
    
    def get_obs(self) -> np.ndarray:
        ...
    
    def save_frame(self):
        if self.n_step < 10:
            save_json(self.curr_frame, PATH_FRAMES / f"frame{self.n_step}.json")
            simplify_iter(self.curr_frame,  PATH_FRAMES / f"frame{self.n_step}_simplify.json")
            too_simplify_iter(self.curr_frame,  PATH_FRAMES / f"frame{self.n_step}_too_simplify.json")
