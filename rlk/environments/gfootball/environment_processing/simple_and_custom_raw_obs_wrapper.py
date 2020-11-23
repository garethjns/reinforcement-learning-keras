from typing import Any, Dict, Union, List, Tuple

import gym
import numpy as np
from gfootball.env.wrappers import Simple115StateWrapper

from rlk.environments.gfootball.environment_processing.raw_obs import RawObs


class SimpleAndCustomRawObsWrapper(gym.Wrapper):
    """

    All:
    dict_keys(['active', 'ball', 'ball_direction', 'ball_owned_player',
              'ball_owned_team', 'ball_rotation', 'designated', 'game_mode',
              'left_team', 'left_team_active', 'left_team_direction', 'left_team_roles',
              'left_team_tired_factor', 'left_team_yellow_card', 'right_team', 'right_team_active',
              'right_team_direction', 'right_team_roles', 'right_team_tired_factor',
              'right_team_yellow_card', 'score', 'steps_left', 'sticky_actions'])

    Processes those apparently not in s115:
    'ball_rotation'

    """

    def __init__(self, env: gym.Env = None):
        """
        :param env: A gym env, or None.
        """
        if env is not None:
            super().__init__(env)

        self.simple_obs_shape = (115,)
        self.raw_obs_shape = (102,)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                                shape=(self.simple_obs_shape[0] + self.raw_obs_shape[0],))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        obs, reward, done, info = self.env.step(action)

        return self.process_obs(obs), reward, done, info

    def reset(self) -> np.ndarray:
        obs = self.env.reset()

        return self.process_obs(obs)

    @staticmethod
    def process_obs(obs: Union[Dict[str, Any], List[Any]]) -> np.ndarray:

        if isinstance(obs, dict):
            # From kaggle env
            obs_for_s115 = obs['players_raw']
            obs_for_raw = obs['players_raw']

        elif isinstance(obs, list):
            # From gfootball Env
            obs_for_s115 = obs
            obs_for_raw = obs[0]

        else:
            raise ValueError("Something unexpected about obs")

        simple_obs = Simple115StateWrapper.convert_observation(obs_for_s115, fixed_positions=False).reshape(-1)
        raw_obs = RawObs.convert_observation(obs_for_raw)

        return np.concatenate([simple_obs, raw_obs.squeeze()])
