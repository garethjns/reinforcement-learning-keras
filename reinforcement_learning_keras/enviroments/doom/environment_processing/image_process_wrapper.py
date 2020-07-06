from typing import Type

import cv2
import gym
import numpy as np


class ImageProcessWrapper(gym.ObservationWrapper):
    """Scale image by given factor and make greyscale."""

    def __init__(self, env: gym.Env, scale: float = 0.4, dtype: Type = np.float32) -> None:
        super().__init__(env)
        self.dtype = dtype
        self.scale = scale

        self._new_size = (int(env.observation_space.shape[0] * scale), int(env.observation_space.shape[1] * scale))
        # New env obs space shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self._new_size, dtype=self.dtype)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return self.process(obs)

    def process(self, frame: np.ndarray) -> np.ndarray:
        gs = np.sum(frame, axis=2) / 3
        gs_resized = cv2.resize(gs, (self._new_size[1], self._new_size[0]), interpolation=cv2.INTER_AREA)
        gs_norm = gs_resized.astype(self.dtype) / 255.0

        return gs_norm
