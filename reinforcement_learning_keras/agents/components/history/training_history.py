from dataclasses import dataclass
from typing import List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from reinforcement_learning_keras.agents.components.history.episode_report import EpisodeReport


@dataclass
class TrainingHistory:
    plotting_on: bool = False
    plot_every: int = 50
    agent_name: str = 'Unnamed agents'
    rolling_average: int = 10

    def __post_init__(self) -> None:
        sns.set()
        self.history: List[EpisodeReport] = []

    def append(self, episode_report: EpisodeReport) -> None:
        self.history.append(episode_report)

    def extend(self, episode_report: List[EpisodeReport]) -> None:
        self.history.extend(episode_report)

    def get_metric(self, metric: str = "total_reward") -> List[Any]:
        return [getattr(ep, metric) for ep in self.history]

    def plot(self, metrics: List[str], show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Plot current history."""
        return self._plot(show=show, metrics=metrics)

    def training_plot(self, show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Plot if it's turned on and is a plot step."""

        if self.plotting_on and (not len(self.history) % self.plot_every):
            return self._plot(show=show, metrics=["total_reward", "frames"])

    def _plot(self, metrics: List[str], show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Create axes and plot. Not storing matplotlib objects to self as they cause pickle issues."""
        plt.close('all')
        fig, axs = plt.subplots(nrows=len(metrics), ncols=1)
        if not isinstance(axs, np.ndarray):
            axs = [axs]

        for ax, metric in zip(axs, metrics):
            ax.plot(self._rolling_average(self.get_metric(metric)), label=metric)
            ax.set_xlabel('N Episodes', fontweight='bold')
            ax.set_ylabel(metric, fontweight='bold')
        axs[0].set_title(self.agent_name, fontweight='bold')

        if show:
            fig.show()

        return fig, ax

    def _rolling_average(self, x) -> np.ndarray:
        """Rolling average over """
        return np.convolve(x, np.ones(self.rolling_average), 'valid') / self.rolling_average

    @property
    def total_frames(self) -> int:
        return int(np.sum(self.get_metric("frames")))

    @property
    def current_performance(self, metric: str = "total_reward") -> float:
        """Return average performance over the last rolling average window."""
        return float(np.mean(self.get_metric(metric)[len(self.history) - self.rolling_average:-1]))
