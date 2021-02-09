import os
from functools import partial
from typing import Any, Dict

from rlk.agents.components.history.training_history import TrainingHistory
from rlk.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from rlk.agents.models.dense_nn import DenseNN
from rlk.agents.models.dense_nn_simple import DenseNNSimple
from rlk.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from rlk.environments.cart_pole.environment_processing.clipepr_wrapper import ClipperWrapper
from rlk.environments.cart_pole.environment_processing.rbf_wrapepr import RBFSWrapper
from rlk.environments.cart_pole.environment_processing.squeeze_wrapper import SqueezeWrapper
from rlk.environments.config_base import ConfigBase


class CartPoleConfig(ConfigBase):
    """Defines config for cart_pole."""
    env_spec = 'CartPole-v0'
    supported_agents = ('actor_critic', 'linear_q', 'dqn', 'double_dqn', 'dueling_dqn', 'double_dueling_dqn',
                        'reinforce', 'random')
    gpu_memory = 128

    @property
    def _default_training_history_kwargs(self) -> Dict[str, Any]:
        return {"plotting_on": self.plot_during_training,
                "plot_every": 25, "rolling_average": 12}

    def _build_for_linear_q(self) -> Dict[str, Any]:
        return {'name': os.path.join(self.folder, 'LinearQAgent'),
                'env_spec': self.env_spec,
                'env_wrappers': [partial(ClipperWrapper, lim=(-1, 1)), RBFSWrapper, SqueezeWrapper],
                'gamma': 0.99,
                'log_exemplar_space': False,
                'final_reward': -200,
                'eps': EpsilonGreedy(eps_initial=0.4, eps_min=0.01, actions_pool=list(range(2)))}

    def _build_for_ac(self):
        return {'name': os.path.join(self.folder, 'ACAgent'),
                'env_spec': self.env_spec,
                "training_history": TrainingHistory(plotting_on=True, plot_every=100, rolling_average=50),
                "model_architecture": DenseNNSimple(output_type='ac', observation_shape=(4,), n_actions=2,
                                                    learning_rate=0.001, unit_scale=8,
                                                    hidden_layer_activations='relu', opt='adam',
                                                    output_activation='softmax'),
                "gamma": 0.99}

    def _build_for_dqn(self) -> Dict[str, Any]:
        return {'name': os.path.join(self.folder, 'DeepQAgent'),
                'env_spec': self.env_spec,
                # 'env_wrappers': [partial(ClipperWrapper, lim=(-100, 100)), RBFSWrapper, SqueezeWrapper],
                'model_architecture': DenseNNSimple(observation_shape=(4,), n_actions=2, opt='adam',
                                                    learning_rate=0.001, unit_scale=8, dueling=False,
                                                    hidden_layer_activations='relu'),
                'gamma': 0.99,
                'final_reward': -1000,
                'replay_buffer_samples': 32,
                'eps': EpsilonGreedy(eps_initial=0.5, decay=0.0002, eps_min=0.002, actions_pool=list(range(2))),
                'replay_buffer': ContinuousBuffer(buffer_size=1000)}

    def _build_for_dueling_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': os.path.join(self.folder, 'DuelingDQN'),
                            'model_architecture': DenseNN(observation_shape=(4,), n_actions=2, opt='adam',
                                                          learning_rate=0.0001, unit_scale=8, dueling=True)})

        return config_dict

    def _build_for_double_dueling_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': os.path.join(self.folder, 'DoubleDuelingDQN'),
                            'double': True,
                            'model_architecture': DenseNN(observation_shape=(4,), n_actions=2, opt='adam',
                                                          learning_rate=0.001, unit_scale=16, dueling=True)})

        return config_dict

    def _build_for_double_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': os.path.join(self.folder, 'DoubleDQN'),
                            'double': True,
                            'model_architecture': DenseNN(observation_shape=(4,), n_actions=2, opt='adam',
                                                          learning_rate=0.0001, unit_scale=16, dueling=False)})

        return config_dict

    def _build_for_reinforce(self) -> Dict[str, Any]:
        return {'name': os.path.join(self.folder, 'REINFORCEAgent'),
                'env_spec': self.env_spec,
                'model_architecture': DenseNN(observation_shape=(4,), n_actions=2, opt='adam', unit_scale=16,
                                              output_activation='softmax', learning_rate=0.001, dueling=False),
                'final_reward': -2,
                'gamma': 0.99,
                'alpha': 0.00001}

    def _build_for_random(self):
        return {'name': os.path.join(self.folder, 'RandomAgent'),
                'env_spec': self.env_spec}
