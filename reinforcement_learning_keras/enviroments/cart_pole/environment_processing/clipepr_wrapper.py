from typing import Tuple

import gym
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reinforcement_learning_keras.enviroments.preprocessing.clipper import Clipper


class ClipperWrapper(gym.ObservationWrapper):
    """Clip all observations to within limits, then StandardScale."""

    def __init__(self, env: gym.Env, lim: Tuple[float, float] = (-1, 1)):
        super().__init__(env)
        # New env obs space shape
        self.observation_space = gym.spaces.Box(low=lim[0], high=lim[1], shape=self.observation_space.shape)

        pipe = Pipeline([('clip', Clipper(lim=lim)),
                         ('ss', StandardScaler())])
        pipe.fit(np.array([[lim[0]] * self.observation_space.shape[0],
                           [lim[1]] * self.observation_space.shape[0]]))
        self.pp = pipe

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return self.pp.transform(obs.reshape(1, -1))
