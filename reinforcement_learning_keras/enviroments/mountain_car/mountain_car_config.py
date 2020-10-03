import os
from typing import Any, Dict

from reinforcement_learning_keras.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from reinforcement_learning_keras.enviroments.cart_pole.models.small_dueling_nn import SmallDuelingNN
from reinforcement_learning_keras.enviroments.config_base import ConfigBase
from reinforcement_learning_keras.enviroments.mountain_car.models.small_nn import SmallNN


class MountainCarConfig(ConfigBase):
    """Defines config for mountain_car"""
    env_spec = 'MountainCar-v0'
    supported_agents = ('linear_q', 'dueling_dqn', 'dqn', 'random')
    gpu_memory = 128

    @property
    def _default_training_history_kwargs(self) -> Dict[str, Any]:
        return {"plotting_on": self.plot_during_training, "plot_every": 200, "rolling_average": 12}

    def _build_for_linear_q(self) -> Dict[str, Any]:
        return {'name': os.path.join(self.folder, 'LinearQAgent'),
                'env_spec': self.env_spec,
                'final_reward': 500,
                'gamma': 0.99,
                'log_exemplar_space': False,
                'eps': EpsilonGreedy(eps_initial=0.3, eps_min=0.005)}

    def _build_for_dqn(self) -> Dict[str, Any]:
        """This isn't tuned."""
        return {'name': os.path.join(self.folder, 'DeepQAgent'),
                'env_spec': self.env_spec,
                'model_architecture': SmallNN(observation_shape=(2,), n_actions=3, output_activation=None,
                                              opt='adam', learning_rate=0.001),
                'gamma': 0.99,
                'final_reward': 650,
                'replay_buffer_samples': 32,
                'eps': EpsilonGreedy(eps_initial=0.1, decay=0.002, eps_min=0.002),
                'replay_buffer': ContinuousBuffer(buffer_size=200)}

    def _build_for_dueling_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': os.path.join(self.folder, 'DuelingDQN'),
                            'model_architecture': SmallDuelingNN(observation_shape=(2,), n_actions=3,
                                                                 opt='adam', learning_rate=0.001)})

        return config_dict

    def _build_for_random(self):
        return {'name': os.path.join(self.folder, 'RandomAgent'),
                'env_spec': self.env_spec}
