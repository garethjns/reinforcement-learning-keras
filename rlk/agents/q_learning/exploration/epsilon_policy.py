from typing import Callable
from dataclasses import dataclass

import numpy as np

from rlk.agents.q_learning.exploration.epsilon_base import EpsilonBase


@dataclass
class EpsilonPolicy(EpsilonBase):
    policy: Callable = lambda s: 0

    def _policy(self, state: np.ndarray) -> int:
        return self.policy(state)
