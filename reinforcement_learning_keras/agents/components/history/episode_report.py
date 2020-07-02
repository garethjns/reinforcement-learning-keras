from dataclasses import dataclass

import numpy as np

from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy


@dataclass
class EpisodeReport:
    frames: int
    time_taken: float
    total_reward: float
    epsilon_used: EpsilonGreedy = None

    def __str__(self) -> str:
        return f"Reward: {self.total_reward} from {self.frames} frames in {self.time_taken} s ({self.fps} f/s). " \
               f"Eps remaining: {self.epsilon_used.eps_current if self.epsilon_used is not None else None}"

    @property
    def fps(self) -> float:
        return np.round(self.frames / self.time_taken)
