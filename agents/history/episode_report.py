from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeReport:
    frames: int
    time_taken: float
    total_reward: float
    epsilon_used: float = np.nan

    def __str__(self) -> str:
        return f"Reward: {self.total_reward} from {self.frames} frames in {self.time_taken} s ({self.fps} f/s)"

    @property
    def fps(self) -> float:
        return np.round(self.frames / self.time_taken)
