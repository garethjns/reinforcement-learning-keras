from dataclasses import dataclass
from functools import reduce
from typing import Union, Callable, Iterable, Dict, Any

import gym


@dataclass
class EnvBuilder:
    env_spec: str
    env_kwargs: Union[None, Dict[str, Any]] = None
    env_wrappers: Iterable[Callable] = None

    def __post_init__(self):
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

        if self._env is None:
            # Make the gym environment and apply the wrappers one by one
            self._env = reduce(lambda inner_env, wrapper: wrapper(inner_env),
                               self.env_wrappers,
                               gym.make(self.env_spec, **self.env_kwargs))

    @property
    def env(self) -> gym.Env:
        """Use to access env, if not ready also makes it ready."""
        self.set_env()

        return self._env
