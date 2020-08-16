from typing import Any, Tuple, Dict, Union
from unittest.mock import MagicMock

import numpy as np


class EnvFixture:
    action_space = MagicMock()
    reward_range = 1
    metadata = None

    def __init__(self, obs_shape: Tuple[int, int, int] = (250, 160)) -> None:
        self.obs_shape = obs_shape
        self._obs = np.ones(obs_shape)
        self._reward = 1.0
        self._stop_in: Union[None, int] = None

        self.observation_space = MagicMock()
        self.observation_space.shape = obs_shape

    def reset(self) -> np.ndarray:
        return self._obs

    def action_indicator(self, action: int) -> None:
        """Called each step, patch to monitor in tests."""
        pass

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        self.action_indicator(action)

        if action == 1:
            # Trigger stop in a few turns
            if self._stop_in is None:
                self._stop_in = 3
            self._stop_in -= 1

        done = False
        info = {}
        reward = self._reward
        obs = self._obs

        if self._stop_in == 0:
            done = True
            obs += 1
            reward += 1

        return obs, reward, done, info
