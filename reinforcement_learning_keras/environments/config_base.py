import abc
from typing import List, Dict, Any

import gym

from reinforcement_learning_keras.agents.components.history.training_history import TrainingHistory


class ConfigBase(abc.ABC):
    env_spec: str
    supported_agents: List[str]
    gpu_memory: int = 256

    def __init__(self, agent_type: str, plot_during_training: bool = True, folder: str = ''):
        self.plot_during_training = plot_during_training
        self.folder = folder

        # Example env
        self.unwrapped_env = gym.make(self.env_spec)
        self.wrapped_env = self.unwrapped_env  # (Default no wrapper)

        self._check_supported(agent_type)
        self.agent_type = agent_type

    def _check_supported(self, agent_type: str):
        if agent_type not in self.supported_agents:
            raise NotImplementedError(f"Agent {agent_type} not in supported agents: {self.supported_agents}")

    @property
    def _default_training_history_kwargs(self) -> Dict[str, Any]:
        return {"plotting_on": self.plot_during_training, "plot_every": 50, "rolling_average": 25}

    def build(self) -> Dict[str, Any]:

        if self.agent_type.lower() == 'linear_q':
            config_dict = self._build_for_linear_q()
        elif self.agent_type.lower() == 'dqn':
            config_dict = self._build_for_dqn()
        elif self.agent_type.lower() == 'dueling_dqn':
            config_dict = self._build_for_dueling_dqn()
        elif self.agent_type.lower() == 'double_dqn':
            config_dict = self._build_for_double_dqn()
        elif self.agent_type.lower() == 'double_dueling_dqn':
            config_dict = self._build_for_double_dueling_dqn()
        elif self.agent_type.lower() == 'reinforce':
            config_dict = self._build_for_reinforce()
        elif self.agent_type.lower() == 'random':
            config_dict = self._build_for_random()
        else:
            raise NotImplementedError

        config_dict.update({'training_history': TrainingHistory(agent_name=config_dict['name'],
                                                                **self._default_training_history_kwargs)})
        return config_dict

    def _build_for_linear_q(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_for_dqn(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_for_dueling_dqn(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_for_double_dqn(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_for_double_dueling_dqn(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_for_random(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_for_reinforce(self) -> Dict[str, Any]:
        raise NotImplementedError
