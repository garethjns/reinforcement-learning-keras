import gym
import numpy as np


class SqueezeWrapper(gym.ObservationWrapper):
    """Remove any extra dimensions (eg. row) that might have been added by wrappers using sklearn components."""
    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.squeeze()
