from functools import partial
from typing import Any, Dict

from agents.components.history.training_history import TrainingHistory
from agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from agents.policy_gradient.loss import reinforce_loss
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
    supported_agents = ('linear_q', 'dueling_dqn', 'dqn', 'reinforce', 'random')
    gpu_memory = 128

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self) -> Dict[str, Any]:
        if self.agent_type.lower() == 'linear_q':
            return self._build_for_linear_q()

        if self.agent_type.lower() == 'dqn':
            return self._build_for_dqn()

        if self.agent_type.lower() == 'dueling_dqn':
            return self._build_for_dueling_dqn()

        if self.agent_type.lower() == 'reinforce':
            return self._build_for_reinforce()

        if self.agent_type.lower() == 'random':
            return self._build_for_random()

    def _build_for_linear_q(self) -> Dict[str, Any]:
        name = 'LinearQAgent'
        return {'name': name,
                'env_spec': self.env_spec,
                'env_wrappers': [partial(ClipperWrapper, lim=(-1, 1)), RBFSWrapper, SqueezeWrapper],
                'gamma': 0.99,
                'log_exemplar_space': False,
                'final_reward': -200,
                'eps': EpsilonGreedy(eps_initial=0.4,
                                     eps_min=0.01),
                'training_history': TrainingHistory(plotting_on=self.plot_during_training,
                                                    plot_every=200,
                                                    rolling_average=12,
                                                    agent_name=name)}

    def _build_for_dqn(self) -> Dict[str, Any]:
        from tensorflow import keras

        name = 'DeepQAgent'

        return {'name': name,
                'env_spec': self.env_spec,
                'env_wrappers': [],
                'model_architecture': SmallNN(observation_shape=(4,), n_actions=2, output_activation=None,
                                              opt=keras.optimizers.Adam(learning_rate=0.001), loss='mse'),
                'gamma': 0.99,
                'learning_rate': 0.0005,
                'final_reward': -10,
                'replay_buffer_samples': 75,
                'eps': EpsilonGreedy(eps_initial=0.05,
                                     decay=0.002,
                                     eps_min=0.002),
                'replay_buffer': ContinuousBuffer(buffer_size=200),
                'training_history': TrainingHistory(plotting_on=self.plot_during_training,
                                                    plot_every=25,
                                                    rolling_average=12,
                                                    agent_name=name)}

    def _build_for_dueling_dqn(self) -> Dict[str, Any]:
        from tensorflow import keras

        name = 'DuelingDQN'
        config_dict = self._build_for_dqn()
        config_dict.update({'name': name,
                            'model_architecture': SmallDuelingNN(observation_shape=(4,), n_actions=2,
                                                                 opt=keras.optimizers.Adam(learning_rate=0.001),
                                                                 loss='mse'),
                            'training_history': TrainingHistory(plotting_on=self.plot_during_training,
                                                                plot_every=25,
                                                                rolling_average=12,
                                                                agent_name=name)})

        return config_dict

    def _build_for_reinforce(self) -> Dict[str, Any]:
        from tensorflow import keras

        name = 'REINFORCEAgent'
        return {'name': name,
                'env_spec': self.env_spec,
                'env_wrappers': [],
                'model_architecture': SmallNN(observation_shape=(4,), n_actions=2, output_activation='softmax',
                                              opt=keras.optimizers.Adam(learning_rate=0.001), loss=reinforce_loss),
                'gamma': 0.99,
                'learning_rate': 0.00001,
                'alpha': 0.00001,
                'training_history': TrainingHistory(plotting_on=self.plot_during_training,
                                                    plot_every=25,
                                                    rolling_average=12,
                                                    agent_name=name)}

    def _build_for_random(self):
        name = 'RandomAgent'
        return {'name': name,
                'env_spec': self.env_spec,
                'training_history': TrainingHistory(plotting_on=self.plot_during_training,
                                                    plot_every=25,
                                                    rolling_average=12,
                                                    agent_name=name)}
