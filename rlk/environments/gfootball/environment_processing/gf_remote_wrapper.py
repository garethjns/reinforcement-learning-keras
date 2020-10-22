import json
from typing import Tuple, List, Dict, Any, Union
from unittest.mock import MagicMock

import gym
import numpy as np
import requests
from asyncio import get_running_loop

class GFRemoteWrapper:
    """
    Create and handle running remote version of the request environment.

    Specific to supported GFootball envs for now. Creates from name and creates local version (not stepped).
    Might be worth adding create from env to make more general.

    Currently does not inherit from gym.Env so needs a few extra things added.
    """

    def __init__(self, env: Union[gym.Env, str], ip: str = "192.168.68.124", port: int = 8000):

        # register_all()

        if isinstance(env, gym.Env):
            env_name = env.spec.id
        else:
            env_name = env

        if env_name is not None:
            if '-v' in env_name:
                self.scenario_name = env_name.split('-')[1]
            else:
                self.scenario_name = env_name
        else:
            self.scenario_name = None
        self.env_name = env_name

        self.ip = ip
        self.port = port
        self.address = f"http://{ip}:{port}"

        self._remote_create = f"{self.address}/create"
        self._remote_reset = f"{self.address}/reset"
        self._remote_step = f"{self.address}/step"
        self._remote_close = f"{self.address}/kill"

        self._create_remote_from_local()
        self.__max_episode_steps = None

    def __repr__(self) -> str:
        return f"GFRemoteWrapper(env={self.env_name}, ip={self.ip}, port={self.port})"

    def _create_remote_from_local(self):
        """
        Create equivalent of local env on remote.

        For now, remote just supports limited set, ie. SMMBuffer(SimpleSMM(create_env('name'))), so
        just passing name.
        """
        resp = requests.get(self._remote_create, params={'env_name': self.env_name})
        self._remote_env_id = resp.json()['id']

    @staticmethod
    def _json_to_obs(resp_obs: str) -> List[np.ndarray]:
        return [np.array(arr) for arr in json.loads(resp_obs)]

    def reset(self) -> List[np.ndarray]:
        resp = requests.get(self._remote_reset,
                            params={'env_id': str(self._remote_env_id)})

        return self._json_to_obs(resp.json()['obs'])

    def step(self, action: int) -> Tuple[List[np.ndarray], float, bool, Dict[str, Any]]:
        resp = requests.get(self._remote_step,
                            params={'env_id': str(self._remote_env_id),
                                    'action': int(action)})
        resp = resp.json()

        return self._json_to_obs(resp['obs']), resp['reward'], resp['done'], resp['info']

    def close(self):
        resp = requests.get(self._remote_close,
                            params={'env_id': str(self._remote_env_id)})

        return resp.json()

    @property
    def _max_episode_steps(self) -> int:
        """This is used by play_episode to set step limit."""
        return self.__max_episode_steps

    @_max_episode_steps.setter
    def _max_episode_steps(self, n_steps: int) -> None:
        # TODO: Send to remote API
        self.__max_episode_steps = n_steps

    @property
    def action_space(self):
        return gym.spaces.Box(low=0, high=19, shape=(1,))

    def render(self):
        pass


if __name__ == "__main__":
    env = GFRemoteWrapper(env_name="GFootball-academy_empty_goal_close-v0")
    obs = env.reset()
    obs, reward, done, info = env.step(1)
    env.close()
