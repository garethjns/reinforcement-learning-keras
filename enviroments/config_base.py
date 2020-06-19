import abc
from typing import List


class ConfigBase(abc.ABC):
    supported_agents: List[str]
    gpu_memory: int = 256
    env_spec: str

    def __init__(self, agent_type: str, plot_during_training: bool = True):
        self.plot_during_training = plot_during_training

        if agent_type not in self.supported_agents:
            raise TypeError(f"Agent {agent_type} not in supported agents: {self.supported_agents}")
        self.agent_type = agent_type

    @abc.abstractmethod
    def build(self):
        pass
