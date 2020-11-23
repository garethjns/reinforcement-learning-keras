import json
from typing import List, Union, Dict, Any, Tuple

import gym
import numpy as np
from gfootball.env.wrappers import Simple115StateWrapper

from rlk.agents.components.helpers.ndarray_encoder import NDArrayEncoder
from rlk.environments.gfootball.environment_processing.raw_obs import RawObs


class SimpleAndRawObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env = None, raw_using: List[str] = None, raw_dump_path: str = None) -> None:
        """
        :param env: A gym env, or None.
        :param raw_using: List of keys to use in raw observations.
        """
        if env is not None:
            super().__init__(env)

        self.raw_dump_path = raw_dump_path
        self.raw_obs = RawObs(using=raw_using)

        self.simple_obs_shape = 115
        self.raw_obs_shape = self.raw_obs.shape[1]

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(self.simple_obs_shape + self.raw_obs_shape,), dtype=np.float32)

    @staticmethod
    def process_obs(obs: Union[Dict[str, Any], List[Any]], using: List[str] = None) -> np.ndarray:
        """Generate array with simple obs and raw obs."""

        if isinstance(obs, dict):
            obs = obs['players_raw']

        simple_obs = Simple115StateWrapper.convert_observation(obs, fixed_positions=False).reshape(-1)
        raw_obs = RawObs(using=using).set_obs(obs[0]).process()

        return np.concatenate([simple_obs, raw_obs.reshape(-1)]) if raw_obs is not None else simple_obs

    def _dump(self, obs: Dict[str, np.ndarray]):
        # Save raw observations to disk, may be used by EpsilonPolicy bot. TODO: This interface should be improved.
        if self.raw_dump_path is not None:
            with open(self.raw_dump_path, 'w') as f:
                json.dump({'players_raw': obs}, f, cls=NDArrayEncoder)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        obs, reward, done, info = self.env.step(action)
        self._dump(obs)

        return self.process_obs(obs, using=self.raw_obs.using), reward, done, info

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._dump(obs)

        return self.process_obs(obs, using=self.raw_obs.using)
