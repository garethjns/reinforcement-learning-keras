from dataclasses import dataclass

import numpy as np

from rlk.agents.q_learning.exploration.epsilon_base import EpsilonBase


@dataclass
class EpsilonGreedy(EpsilonBase):
    def _policy(self, state: np.ndarray) -> int:
        return np.random.choice(self.actions_pool)
