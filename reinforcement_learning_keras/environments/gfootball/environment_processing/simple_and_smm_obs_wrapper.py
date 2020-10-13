from typing import Tuple, Dict, Any, List, Union

import gym
import numpy as np
from gfootball.env import observation_preprocessing
from gfootball.env.wrappers import Simple115StateWrapper


class SimpleAndSMMObsWrapper(gym.Wrapper):
    """
    This wrapper is designed to accept obs from either the unwrapped GFootballEnv using the gym interface, or the Kaggle
    running in Kaggle environments. See process_obs for notes on the differences.

    From the "raw" observations it generates both the SMM and Simple115State representations. There are returned as
    as Tuple[smm_obs, simple_obs].
    """

    def __init__(self, env: gym.Env = None):
        """
        :param env: A gym env, or None.
        """
        if env is not None:
            super().__init__(env)

        self.simple_obs_shape = (115,)  # TODO: It's possible this can be bigger

        self.observation_space = gym.spaces.Tuple(
            [gym.spaces.Box(low=0, high=255, shape=(72, 96, 4), dtype=np.uint8),
             gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.simple_obs_shape, dtype=np.float32)])

    @staticmethod
    def process_obs(obs: Union[Dict[str, Any], List[Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obs can be from gym env or the version passed from Kaggle runner.

        We need to extract this dict to generate simple and SMM obs:
        dict_keys(['left_team_tired_factor', 'left_team_yellow_card', 'right_team_tired_factor', 'left_team',
                    'ball_owned_player', 'right_team_yellow_card', 'ball_rotation', 'ball_owned_team', 'ball',
                    'right_team_roles', 'right_team_active', 'steps_left', 'score', 'right_team', 'left_team_roles',
                    'ball_direction', 'left_team_active', 'left_team_direction', 'right_team_direction', 'game_mode',
                    'designated', 'active', 'sticky_actions'])

        Which is located in:
         - Kag obs: obs_kag_env['players_raw'][0].keys():
         - Gym obs: obs_gym_env[0].keys()
        """

        if isinstance(obs, dict):
            obs = obs['players_raw']

        # This can return multiple rows when env has:
        # number_of_left_players_agent_controls=1 and number_of_right_players_agent_controls=1
        simple_obs = Simple115StateWrapper.convert_observation(obs, fixed_positions=False).reshape(-1)
        smm_obs = observation_preprocessing.generate_smm([obs[0]])

        return smm_obs, simple_obs

    def step(self, action: int) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict[Any, Any]]:
        obs, reward, done, info = self.env.step(action)

        return self.process_obs(obs), reward, done, info

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        obs = self.env.reset()
        return self.process_obs(obs)
