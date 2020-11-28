from typing import Any, Dict, Union, List

import numpy as np


class RawObs:
    """
    All:
    dict_keys(['active', 'ball', 'ball_direction', 'ball_owned_player', 'ball_owned_team', 'ball_rotation',
               'designated', 'game_mode', 'left_team', 'left_team_active', 'left_team_direction', 'left_team_roles',
               'left_team_tired_factor', 'left_team_yellow_card', 'right_team', 'right_team_active',
               'right_team_direction', 'right_team_roles', 'right_team_tired_factor',
               'right_team_yellow_card', 'score', 'steps_left', 'sticky_actions'])

    Default excludes (based on https://www.kaggle.com/denisvodchyts/dqn-tf-agent-with-rule-base-collection-policy):
    'left_team_active', 'right_team_active', 'designated', 'left_team_tired_factor', 'left_team_roles',
    'left_team_yellow_card', 'right_team_roles', 'right_team_yellow_card',  'right_team_tired_factor'
    'steps_left'

    Processes those apparently not in s115:
    'ball_rotation'
    """
    data: Dict[str, Any]
    standard_keys = ['active', 'ball', 'ball_direction', 'ball_owned_player',
                     'ball_owned_team', 'ball_rotation', 'designated', 'game_mode',
                     'left_team', 'left_team_active', 'left_team_direction', 'left_team_roles',
                     'left_team_tired_factor', 'left_team_yellow_card', 'right_team', 'right_team_active',
                     'right_team_direction', 'right_team_roles', 'right_team_tired_factor',
                     'right_team_yellow_card', 'score', 'steps_left', 'sticky_actions']

    active_n: int = 1
    ball_n: int = 3
    ball_direction_n: int = 3
    ball_owned_player_n: int = 1
    ball_owned_team_n: int = 1
    ball_rotation_n: int = 3
    designated_n: int = 1
    game_mode_n: int = 1
    left_team_n: int = 22
    left_team_active_n: int = 11
    left_team_direction_n: int = 22
    left_team_roles_n: int = 11
    left_team_tired_factor_n: int = 11
    left_team_yellow_card_n: int = 11
    right_team_n: int = 22
    right_team_active_n: int = 11
    right_team_direction_n: int = 22
    right_team_roles_n: int = 11
    right_team_tired_factor_n: int = 11
    right_team_yellow_card_n: int = 11
    score_n: int = 2
    steps_left_n: int = 1
    sticky_actions_n: int = 10

    # Custom
    distance_to_ball_n: int = 22

    def __init__(self, using: List[str] = None, using_custom: List[str] = None) -> None:

        if using is None:
            using = ['active', 'ball', 'ball_direction', 'ball_owned_player', 'ball_owned_team', 'ball_rotation',
                     'game_mode', 'left_team', 'left_team_direction', 'right_team', 'right_team_direction', 'score',
                     'sticky_actions']
        self.using = using

        if using_custom is None:
            using_custom = []
        self.using_custom = using_custom

        self.shape = (1, sum([getattr(self, f"{key}_n") for key in self.using + self.using_custom]))

    def set_obs(self, data: Union[List[Dict[str, Any]], Dict[str, Any]]):
        if isinstance(data, list):
            self.data = data[0]
        else:
            self.data = data

        return self

    @staticmethod
    def _euclidean_distance(x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _add_distance_to_ball(self) -> np.ndarray:
        left_team = self._euclidean_distance(x1=np.array(self.data['ball'])[0],
                                             x2=np.array(self.data["left_team"])[:, 0],
                                             y1=np.array(self.data['ball'])[1],
                                             y2=np.array(self.data["left_team"])[:, 1])
        right_team = self._euclidean_distance(x1=np.array(self.data['ball'])[0],
                                              x2=np.array(self.data["right_team"])[:, 0],
                                              y1=np.array(self.data['ball'])[1],
                                              y2=np.array(self.data["right_team"])[:, 1])

        return np.expand_dims(np.concatenate([left_team, right_team]), axis=0)

    def process_key(self, key: str) -> np.ndarray:
        raw_obs = np.array(self.data[key])
        if len(raw_obs.shape) == 0:
            raw_obs = np.expand_dims(raw_obs, axis=0)

        if len(raw_obs.shape) > 1:
            raw_obs = raw_obs.flatten()

        return np.expand_dims(raw_obs, axis=0)

    def process(self) -> Union[np.ndarray, None]:
        obs = [self.process_key(key) for key in self.using + self.using_custom]

        return np.concatenate(obs, axis=1) if len(obs) > 0 else None

    @classmethod
    def convert_observation(cls, data: Union[List[Dict[str, Any]], Dict[str, Any]],
                            using: List[str] = None) -> np.ndarray:
        return cls(using=using).set_obs(data).process()
