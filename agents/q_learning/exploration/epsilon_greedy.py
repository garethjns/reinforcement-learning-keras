from dataclasses import dataclass
from typing import Callable, Any

import numpy as np


@dataclass
class EpsilonGreedy:
    """Handles epsilon-greedy action selection, decay of epsilon during training."""
    eps_initial: float = 0.2
    decay: float = 0.0001
    decay_schedule: str = 'compound'
    eps_min: float = 0.01
    state = None

    def __post_init__(self) -> None:
        self.eps_current = self.eps_initial

        valid_decay = ('linear', 'compound')
        if self.decay_schedule.lower() not in valid_decay:
            raise ValueError(f"Invalid decay schedule {self.decay_schedule}. Pick from {valid_decay}.")

        self._set_random_state()

    def _set_random_state(self) -> None:
        self._state = np.random.RandomState(self.state)

    def _linear_decay(self) -> float:
        return self.eps_current - self.decay

    def _compound_decay(self) -> float:
        return self.eps_current - self.eps_current * self.decay

    def _decay(self):
        new_eps = np.nan
        if self.decay_schedule.lower() == 'linear':
            new_eps = self._linear_decay()

        if self.decay_schedule.lower() == 'compound':
            new_eps = self._compound_decay()

        return max(self.eps_min, new_eps)

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
            self.eps_current = self._decay()
            if self._state.random() < self.eps_current:
                return random_option()

        return greedy_option()

    @classmethod
    def future_value(cls, eps: float, decay: float, steps: int, decay_schedule: str) -> float:
        """
        Calculate what eps will be after a number of steps.

        Decay is either decay% per step (like compound interest), or linear with constant decay amount.

        :param eps: Current/initial epsilon.
        :param decay: Decay rate per step.
        :param steps: Number of steps (usually training frames, rather than whole episodes)
        :param decay_schedule: 'linear' or 'compound'.
        :return:
        """
        if decay_schedule.lower() == 'compound':
            return eps * (1 - decay) ** steps

        if decay_schedule.lower() == 'linear':
            return eps - decay * steps
