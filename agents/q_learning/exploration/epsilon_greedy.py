from dataclasses import dataclass
from typing import Callable, Any

import numpy as np


@dataclass
class EpsilonGreedy:
    """Handles epsilon-greedy action selection, decay of epsilon during training."""
    eps_initial: float = 0.2
    decay: float = 0.0001
    eps_min: float = 0.01
    state = None

    def __post_init__(self) -> None:
        self.eps_current = self.eps_initial
        self._set_random_state()

    def _set_random_state(self) -> None:
        self._state = np.random.RandomState(self.state)

    def select(self, greedy_option: Callable, random_option: Callable,
               training: bool = False) -> Any:
        """
        Apply epsilon greedy selection.

        If training, decay epsilon, and return selected option. If not training, just return greedy_option.

        Use of lambdas is to avoid unnecessarily picking between two pre-computed options.

        :param greedy_option: Function to evaluate if random option is NOT picked.
        :param random_option: Function to evaluate if random option IS picked.
        :param training: Bool indicating if call is during training and to use epsilon greedy and decay.
        :return: Evaluated selected option.
        """
        if training:
            self.eps_current = max(self.eps_min, self.eps_current - self.eps_current * self.decay)
            if self._state.random() < self.eps_current:
                return random_option()

        return greedy_option()
