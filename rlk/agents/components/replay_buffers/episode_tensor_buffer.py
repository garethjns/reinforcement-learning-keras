import collections
from typing import Deque, List

import numpy as np
import tensorflow as tf

from rlk.agents.components.replay_buffers.replay_buffer_base import ReplayBufferBase


class EpisodeTensorBuffer(ReplayBufferBase):
    """
    Buffer for a single episode.

    Values from models need to be kept as tf.Tensor, to keep track of gradients.
    """
    _action_prob_buffer: Deque[tf.Tensor]
    _critic_value_buffer: Deque[tf.Tensor]
    _rewards_buffer: Deque[float]

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.clear()

    def __len__(self) -> int:
        return len(self._action_prob_buffer)

    def clear(self) -> None:
        self._action_prob_buffer = collections.deque(maxlen=self.max_size)
        self._critic_value_buffer = collections.deque(maxlen=self.max_size)
        self._rewards_buffer = collections.deque(maxlen=self.max_size)

    def append(self, action_prob: tf.Tensor, critic_value: tf.Tensor, reward: float) -> None:
        self._action_prob_buffer.append(action_prob)
        self._critic_value_buffer.append(critic_value)
        self._rewards_buffer.append(reward)

    @staticmethod
    def _calc_discounted_rewards(rr: List[float], gamma: float = 0.99) -> np.ndarray:
        """Calculate discounted rewards for a whole episode and normalise."""

        # Full episode returns
        disc_rr = np.zeros_like(rr)
        cumulative_reward = 0
        for t in reversed(range(0, disc_rr.size)):
            cumulative_reward = cumulative_reward * gamma + rr[t]
            disc_rr[t] = cumulative_reward

        # Normalise
        disc_rr_mean = np.mean(disc_rr)
        disc_rr_std = np.std(disc_rr) + 1e-9
        disc_rr_norm = (disc_rr - disc_rr_mean) / disc_rr_std

        return np.vstack(disc_rr_norm)

    def get_discounted_rewards(self, gamma: float = 0.99) -> np.array:
        return self._calc_discounted_rewards(list(self._rewards_buffer), gamma=gamma)

    def get_critic_values(self) -> List[tf.Tensor]:
        return list(self._critic_value_buffer)

    def get_action_probs(self) -> List[tf.Tensor]:
        return list(self._action_prob_buffer)
