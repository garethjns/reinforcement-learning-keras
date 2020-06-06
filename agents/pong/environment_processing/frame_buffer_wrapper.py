import collections
from typing import Any, Dict, Tuple

import gym
import numpy as np


class FrameBufferWrapper(gym.Wrapper):
    """
    Adds last step obs to buffer, returns whole buffer.

    Returned buffer contains previous steps, eg. for env.step at t=3, returns obs for t=3, t=2, t=1. In games like pong,
    this adds directional information to the returned observations.
    """

    def __init__(self, env: gym.Env,
                 obs_shape: Tuple[int, int] = (84, 84),
                 buffer_length: int = 3,
                 buffer_function: str = 'stack') -> None:
        """
        :param env: Gym env.
        :param obs_shape: Expected shape of single observation.
        :param buffer_length: Number of frames to include in buffer.
        :param buffer_function: Function to apply to use contents of buffer. Supports 'stack or 'diff':
                                  - 'stack' stack contents of buffer on a new (final) axis
                                  - 'diff' take diff between two frames without changing dimensions.
        """
        super().__init__(env)
        self._buffer_length = buffer_length
        self._buffer_function = buffer_function
        self._obs_shape = obs_shape
        self._prepare_obs_buffer()

    def _prepare_obs_buffer(self) -> None:
        """Create buffer and preallocate with empty arrays of expected shape."""

        self._obs_buffer = collections.deque(maxlen=self._buffer_length)

        for _ in range(self._buffer_length):
            self._obs_buffer.append(np.zeros(shape=self._obs_shape))

    def _buffer_obs(self) -> np.ndarray:
        agg_buff = None

        if self._buffer_function == "stack":
            agg_buff = np.stack([obs.squeeze() for obs in self._obs_buffer],
                                axis=len(self._obs_shape))

        if self._buffer_function == 'diff':
            if self._buffer_length != 2:
                raise ValueError("When using diff, buffer length must be 2.")
            agg_buff = self._obs_buffer[1].squeeze() - self._obs_buffer[0].squeeze()
            agg_buff = np.expand_dims(agg_buff,
                                      axis=len(self._obs_shape))

        if agg_buff is None:
            raise ValueError(f"Unknown buffer op {self._buffer_function}")

        return agg_buff

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """Step env, add new obs to buffer, return buffer."""
        obs, reward, done, info = self.env.step(action)
        self._obs_buffer.append(obs)

        return self._buffer_obs(), reward, done, info

    def reset(self) -> np.ndarray:
        """Add initial obs to end of pre-allocated buffer.

        :return: Buffered observation
        """
        self._prepare_obs_buffer()
        obs = self.env.reset()
        self._obs_buffer.append(obs)

        return self._buffer_obs()
