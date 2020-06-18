import collections
from typing import Any, Dict, Tuple, Callable

import gym
import numpy as np


class MaxAndSkipWrapper(gym.Wrapper):
    """Pools frames across steps, returning max, and repeat action for a number of frames."""

    def __init__(self, env: gym.Env,
                 frame_buffer_length: int = 2,
                 n_action_frames: int = 4,
                 frame_buffer_agg_f: Callable = np.max) -> None:
        """

        :param env: Gym environment to wrap, "inner environment".
        :param frame_buffer_length: Max number of frames to collect. FIFO buffer, contains most recent frames only.
        :param n_action_frames: Number of frames to repeat action for. Can be longer than frame buffer.
        :param frame_buffer_agg_f: Function used to aggregate frames in frame buffer to create observation.
                                   Default np.max.
        """
        super().__init__(env)
        self._n_action_frames = n_action_frames
        self._frame_buffer_length = frame_buffer_length
        self._frame_buffer_agg_f = frame_buffer_agg_f
        self._prepare_frame_buffer()

    def _prepare_frame_buffer(self) -> None:
        self._frame_buffer = collections.deque(maxlen=self._frame_buffer_length)

    def _aggregate_buffer_frames(self, ) -> np.ndarray:
        return self._frame_buffer_agg_f(np.stack(self._frame_buffer), axis=0)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """
        Each step call iterates inner env 4 times, max of these is returned.

        Same action is applied for each step, and reward accumulated. If done, loop breaks and returns outputs
        based on shorter pool.

        :param action: Int id of action to perform.
        """
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self._n_action_frames):
            obs, reward, done, info = self.env.step(action)
            self._frame_buffer.append(obs)
            total_reward += reward
            if done:
                break

        agg_frame = self._aggregate_buffer_frames()

        return agg_frame, total_reward, done, info

    def reset(self) -> np.ndarray:
        """
        Additionally clears frame buffer

        :return: Observation from inner_env.reset() call.
        """
        self._frame_buffer.clear()
        obs = self.env.reset()
        self._frame_buffer.append(obs)

        return obs
