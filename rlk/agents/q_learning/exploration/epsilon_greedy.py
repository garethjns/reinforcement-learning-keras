import copy
from dataclasses import dataclass
from typing import Callable, Any, List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EpsilonGreedy:
    """
    Handles epsilon-greedy action selection, decay of epsilon during training.

    Decay schedule can be linear or compound, with additional perturbation by readding epsilon every n steps.

    .future_value can be used to calculate decay for different step, rate, and schedule combinations. It ignores
    other factors (min eps and any perurbation).

     .simulate runs epsilon and returns complete output with current settings, from current point.

     There's no protection against creating an object that increases epsilon, so be careful....

    Examples
    1) Simple linear decay to min_eps
    >>> eps = EpsilonGreedy(eps_initial=1, decay=0.0002, decay_schedule='linear')

    2) Simple compound decay to mins_eps
    >>> eps = EpsilonGreedy(eps_initial=1, decay=0.00075, decay_schedule='compound')

    3) Compund decay, perturb by 0.5 when min_eps is reeached
    >>> eps = EpsilonGreedy(eps_initial=1, decay=0.001, decay_schedule='compound',
    >>>                     perturb_increase_every=3000, perturb_increase_mag=0.5)

    4) Compound decay, spends time at min_eps
    >>> eps = EpsilonGreedy(eps_initial=1, decay=0.01, decay_schedule='compound',
    >>>                    perturb_increase_every=1000, perturb_increase_mag=0.5)

    :param eps_initial: Initial epsilon value.
    :param decay: Decay rate in percent (should be positive to decay).
    :param decay_schedule: 'linear' or 'compound'.
    :param eps_min: The min value epsilon can fall to.
    :param state: Random state, used to pick between the greedy or random options.
    :param perturb_increase_every: Increase epsilon every n steps. Defualt 0.
    :param perturb_increase_mag: Value to add every perturb_increase_every. Default 0.
    """
    eps_initial: float = 0.2
    decay: float = 0.0001
    decay_schedule: str = 'compound'
    eps_min: float = 0.01
    state = None
    perturb_increase_every: int = 0
    perturb_increase_mag: float = 0

    def __post_init__(self) -> None:
        self._step: int = 0
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

        if (self.perturb_increase_every > 0) and (self._step > 0) and (not self._step % self.perturb_increase_every):
            new_eps += self.perturb_increase_mag

        self._step += 1

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

    def simulate(self, steps: int = 10000, plot: bool = False) -> List[float]:
        eps = copy.copy(self)

        eps_value = []
        for _ in range(steps):
            eps.select(lambda: 0, lambda: 1, training=True)
            eps_value.append(eps.eps_current)

        if plot:
            plt.plot(eps_value)
            plt.xlabel('Step')
            plt.ylabel('Epsilon')
            plt.show()

        return eps_value

    @classmethod
    def future_value(cls, eps: float, decay: float, steps: int, decay_schedule: str) -> float:
        """
        Calculate what eps will be after a number of steps.

        Decay is either decay
        % per step (like compound interest), or linear with constant decay amount.

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
