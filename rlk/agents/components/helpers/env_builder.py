from functools import reduce
from typing import Union, Callable, Iterable, Dict, Any

import gym


class EnvBuilder:
    def __init__(self, env_spec: str, env_kwargs: Union[None, Dict[str, Any]] = None,
                 env_wrappers: Iterable[Callable] = None, remote: bool = False,
                 ip: Union[None, str] = None, port: int = 8000):

        self.env_spec = env_spec
        self.env_kwargs = env_kwargs
        self.env_wrappers = env_wrappers
        self.remote = remote
        self.ip = ip
        self.port = port

        self._env: Union[None, gym.Env] = None
        if self.env_wrappers is None:
            self.env_wrappers = []

        if self.env_kwargs is None:
            self.env_kwargs = {}

        self._register_other_envs()
        self.set_env()

    @staticmethod
    def _register_other_envs() -> None:
        """Try and import supported envs (may not be installed). Need to do this to register them with gym."""
        try:
            import vizdoomgym
        except ImportError:
            pass

    def set_env(self, env: Union[None, gym.Env] = None) -> None:
        """
        Create a new env object from the spec, or set a new one.

        Can specify a new env, this is useful, for example, to add a Monitor wrapper.
        """
        if env is not None:
            self._env = env
        else:
            if self._env is None:
                if self.remote:
                    # No general remote wrapper yet, this is the only one
                    from rlk.environments.gfootball.environment_processing.gf_remote_wrapper import GFRemoteWrapper
                    self._env = GFRemoteWrapper(self.env_spec, ip=self.ip, port=self.port)
                else:
                    # Make the gym environment and apply the wrappers one by one
                    self._env = reduce(lambda inner_env, wrapper: wrapper(inner_env),
                                       self.env_wrappers,
                                       gym.make(self.env_spec, **self.env_kwargs))

    @property
    def env(self) -> gym.Env:
        """Use to access env, if not ready also makes it ready."""
        self.set_env()

        return self._env
