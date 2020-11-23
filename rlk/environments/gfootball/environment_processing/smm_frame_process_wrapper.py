import collections
from typing import Any, Dict, Tuple, Union

import gym
import numpy as np


class SMMFrameProcessWrapper(gym.Wrapper):
    """
    Wrapper for processing frames from SMM observation wrapper from football env.

    Input is (72, 96, 4), where last dim is (team 1 pos, team 2 pos, ball pos, active player pos). Range 0 -> 255.
    Output is (72, 96, 4) as difference to last frame for all. Range -1 -> 1

    If input is Tuple, assumes SMM input is index [0] and only buffers that.
    """

    def __init__(self, env: Union[None, gym.Env] = None,
                 obs_shape: Tuple[int, int] = (72, 96, 4)) -> None:
        """
        :param env: Gym env.
        :param obs_shape: Expected shape of single observation.
        """
        if env is not None:
            super().__init__(env)
        else:
            self.env = None

        self._buffer_length = 2
        self._obs_shape = obs_shape
        self._prepare_obs_buffer()

    def __repr__(self) -> str:
        if self.env is None:
            return f"RemoteWrapper(env=None, obs_shape={self._obs_shape})"
        else:
            return super().__repr__()

    @staticmethod
    def _normalise_frame(frame: np.ndarray) -> np.ndarray:
        return frame / 255.0

    def _prepare_obs_buffer(self) -> None:
        """Create buffer and preallocate with empty arrays of expected shape."""

        self._obs_buffer = collections.deque(maxlen=self._buffer_length)

        for _ in range(self._buffer_length):
            self._obs_buffer.append(np.zeros(shape=self._obs_shape))

    def build_buffered_obs(self) -> np.ndarray:
        agg_buff = np.empty(self._obs_shape)
        for f in range(self._obs_shape[-1]):
            agg_buff[..., f] = self._obs_buffer[1][..., f] - self._obs_buffer[0][..., f]

        return agg_buff

    @staticmethod
    def _split_obs(obs: Union[np.ndarray, Tuple[np.ndarray, Any]]) -> Tuple[np.ndarray, Union[None, Any]]:
        if isinstance(obs, Tuple):
            smm_obs = obs[0]
            other_obs = obs[1::]
        else:
            smm_obs = obs
            other_obs = None

        return smm_obs, other_obs

    @staticmethod
    def _rejoin_obs(smm_obs: np.ndarray, other_obs: Union[None, Any]) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        if other_obs is None:
            return smm_obs
        else:
            return tuple([smm_obs] + list(other_obs))

    def process(self, obs: Union[np.ndarray, Tuple[np.ndarray, Any]]) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        smm_obs, other_obs = self._split_obs(obs)
        smm_obs = self._normalise_frame(smm_obs)
        self._obs_buffer.append(smm_obs)
        smm_obs_buff = self.build_buffered_obs()
        joined_obs = self._rejoin_obs(smm_obs_buff, other_obs)

        return joined_obs

    def step(self, action: int) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, Any]],
                                         float, bool, Dict[Any, Any]]:
        """Step env, add new obs to buffer, return buffer."""
        obs, reward, done, info = self.env.step(action)

        return self.process(obs), reward, done, info

    def reset(self) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """Add initial obs to end of pre-allocated buffer.

        :return: Buffered observation
        """
        self._prepare_obs_buffer()
        obs = self.env.reset()

        return self.process(obs)
