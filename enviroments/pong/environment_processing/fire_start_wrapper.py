from typing import Any, Dict, Tuple

import gym
import numpy as np


class FireStartWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, fire_action_id: int = 1):
        super().__init__(env)
        self.fire_action_id = fire_action_id

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        return self.env.step(action)

    def reset(self) -> np.ndarray:
        self.env.reset()
        obs, _, done, _ = self.env.step(self.fire_action_id)

        return obs
