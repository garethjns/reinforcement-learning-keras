from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass()
class TrainingHistory:
    plotting_on: bool = False
    plot_every: int = 50
    agent_name: str = 'Unnamed agents'
    rolling_average: int = 10

    def __post_init__(self):
        sns.set()
        self.history: List[float] = []

    def append(self, episode_reward: float):
        self.history.append(episode_reward)

    def extend(self, episode_rewards: List[float]):
        self.history.extend(episode_rewards)

    def plot(self, show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Plot current history."""
        return self._plot(show=show)

    def training_plot(self, show: bool = True):
        """Plot if it's turned on and is a plot step."""

        if self.plotting_on and (not len(self.history) % self.plot_every):
            self._plot(show=show)

    def _plot(self, show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Create axes and plot. Not storing matplotlib objects to self as they cause pickle issues."""
        plt.close('all')
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(self._rolling_average(self.history))
        ax.set_title(self.agent_name, fontweight='bold')
        ax.set_xlabel('N Episodes', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')

        if show:
            fig.show()

        return fig, ax

    def _rolling_average(self, x) -> np.ndarray:
        """Rolling average over """
        return np.convolve(x, np.ones(self.rolling_average), 'valid') / self.rolling_average

    @property
    def current_performance(self) -> float:
        """Return average performance over the last rolling average window."""
        return float(np.mean(self.history[len(self.history) - self.rolling_average:-1]))
