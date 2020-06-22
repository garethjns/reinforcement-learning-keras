from functools import partial
from typing import Any, Dict

from agents.components.history.training_history import TrainingHistory
from agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from enviroments.cart_pole.environment_processing.clipepr_wrapper import ClipperWrapper
from enviroments.cart_pole.environment_processing.rbf_wrapepr import RBFSWrapper
from enviroments.cart_pole.environment_processing.squeeze_wrapper import SqueezeWrapper
from enviroments.cart_pole.models.small_dueling_nn import SmallDuelingNN
from enviroments.cart_pole.models.small_nn import SmallNN
from enviroments.config_base import ConfigBase


class CartPoleConfig(ConfigBase):
    """Defines config for cart_pole."""
    env_spec = 'CartPole-v0'
    supported_agents = ('linear_q', 'dqn', 'double_dqn', 'dueling_dqn', 'double_dueling_dqn', 'reinforce', 'random')
    gpu_memory = 128

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self) -> Dict[str, Any]:
        config_dict: Dict[str, Any] = {}

        if self.agent_type.lower() == 'linear_q':
            config_dict = self._build_for_linear_q()

        if self.agent_type.lower() == 'dqn':
            config_dict = self._build_for_dqn()

        if self.agent_type.lower() == 'dueling_dqn':
            config_dict = self._build_for_dueling_dqn()

        if self.agent_type.lower() == 'double_dqn':
            config_dict = self._build_for_double_dqn()

        if self.agent_type.lower() == 'double_dueling_dqn':
            config_dict = self._build_for_double_dueling_dqn()

        if self.agent_type.lower() == 'reinforce':
            config_dict = self._build_for_reinforce()

        if self.agent_type.lower() == 'random':
            config_dict = self._build_for_random()

        config_dict.update({'training_history': TrainingHistory(plotting_on=self.plot_during_training,
                                                                plot_every=25, rolling_average=12,
                                                                agent_name=config_dict['name'])})
        return config_dict

    def _build_for_linear_q(self) -> Dict[str, Any]:
        return {'name': 'LinearQAgent',
                'env_spec': self.env_spec,
                'env_wrappers': [partial(ClipperWrapper, lim=(-1, 1)), RBFSWrapper, SqueezeWrapper],
                'gamma': 0.99,
                'log_exemplar_space': False,
                'final_reward': -200,
                'eps': EpsilonGreedy(eps_initial=0.4, eps_min=0.01)}

    def _build_for_dqn(self) -> Dict[str, Any]:
        return {'name': 'DeepQAgent',
                'env_spec': self.env_spec,
                'model_architecture': SmallNN(observation_shape=(4,), n_actions=2, output_activation=None,
                                              opt='adam', learning_rate=0.001),
                'gamma': 0.99,
                'final_reward': -200,
                'replay_buffer_samples': 75,
                'eps': EpsilonGreedy(eps_initial=0.2, decay=0.002, eps_min=0.002),
                'replay_buffer': ContinuousBuffer(buffer_size=200)}

    def _build_for_dueling_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': 'DuelingDQN',
                            'model_architecture': SmallDuelingNN(observation_shape=(4,), n_actions=2, opt='adam',
                                                                 learning_rate=0.001)})

        return config_dict

    def _build_for_double_dueling_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': 'DoubleDuelingDQN',
                            'double': True,
                            'model_architecture': SmallDuelingNN(observation_shape=(4,), n_actions=2, opt='adam',
                                                                 learning_rate=0.001)})

        return config_dict

    def _build_for_double_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': 'DoubleDQN',
                            'double': True,
                            'model_architecture': SmallNN(observation_shape=(4,), n_actions=2, opt='adam',
                                                          learning_rate=0.001)})

        return config_dict

    def _build_for_reinforce(self) -> Dict[str, Any]:
        return {'name': 'REINFORCEAgent',
                'env_spec': self.env_spec,
                'model_architecture': SmallNN(observation_shape=(4,), n_actions=2, output_activation='softmax',
                                              opt='adam', learning_rate=0.001),
                'final_reward': -2,
                'gamma': 0.99,
                'alpha': 0.00001}

    def _build_for_random(self):
        return {'name': 'RandomAgent',
                'env_spec': self.env_spec}
