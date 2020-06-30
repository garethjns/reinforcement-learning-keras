import gym
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion


class RBFSWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        # Sample observations from env and fit pipeline
        obs = np.array([self.env.observation_space.sample() for _ in range(10000)])
        rbfs = FeatureUnion([('rbf0', RBFSampler(gamma=100, n_components=60)),
                             ('rbf1', RBFSampler(gamma=1, n_components=60)),
                             ('rbf2', RBFSampler(gamma=0.02, n_components=60))])
        trans_obs = rbfs.fit_transform(obs)
        self.rbfs = rbfs

        # New env obs space shape
        self.observation_space = gym.spaces.Box(low=trans_obs.min() * 2,
                                                high=trans_obs.max() * 2,
                                                shape=(180,))

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return self.rbfs.transform(obs.reshape(1, -1))
