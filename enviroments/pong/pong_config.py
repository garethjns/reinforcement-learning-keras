from functools import partial
from typing import Any, Dict

import gym

from agents.components.history.training_history import TrainingHistory
from agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from enviroments.config_base import ConfigBase
from enviroments.pong.environment_processing.fire_start_wrapper import FireStartWrapper
from enviroments.pong.environment_processing.frame_buffer_wrapper import FrameBufferWrapper
from enviroments.pong.environment_processing.image_process_wrapper import ImageProcessWrapper
from enviroments.pong.environment_processing.max_and_skip_wrapper import MaxAndSkipWrapper
from enviroments.pong.models.conv_nn import ConvNN

# Wrappers as used by models
PONG_ENV_SPEC = "PongNoFrameskip-v4"
PONG_ENV = gym.make("PongNoFrameskip-v4")

PONG_WRAPPERS_STACK = [MaxAndSkipWrapper, ImageProcessWrapper, FireStartWrapper, FrameBufferWrapper]
PONG_WRAPPERS_DIFF = [MaxAndSkipWrapper, ImageProcessWrapper, FireStartWrapper,
                      partial(FrameBufferWrapper, buffer_length=2, buffer_function='diff')]

PONG_ENV_STACK = FrameBufferWrapper(FireStartWrapper(ImageProcessWrapper(MaxAndSkipWrapper(PONG_ENV))))
PONG_ENV_DIFF = FrameBufferWrapper(FireStartWrapper(ImageProcessWrapper(MaxAndSkipWrapper(PONG_ENV))),
                                   buffer_length=2, buffer_function='diff')


class PongConfig(ConfigBase):
    """Defines configs for Pong."""
    env_spec = 'PongNoFrameskip-v4'
    supported_agents = ('dqn', 'random')
    supported_modes = ('diff', 'stack')
    gpu_memory: int = 4096

    def __init__(self, mode: str = 'diff', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if mode not in self.supported_modes:
            raise ValueError(f"Mode {mode} is not a supported mode ({self.supported_modes})")

        if mode == "diff":
            self.env_wrappers = PONG_WRAPPERS_DIFF
            self.frame_depth = 1
        if mode == "stack":
            self.env_wrappers = PONG_WRAPPERS_STACK
            self.frame_depth = 3

    def build(self) -> Dict[str, Any]:

        if self.agent_type.lower() == 'dqn':
            return self._build_for_dqn()

        if self.agent_type.lower() == 'random':
            return self._build_for_random()

    def _build_for_dqn(self) -> Dict[str, Any]:
        from tensorflow import keras

        name = 'DeepQAgent'

        return {'name': name,
                'env_spec': self.env_spec,
                'env_wrappers': self.env_wrappers,
                'model_architecture': ConvNN(observation_shape=(84, 84, self.frame_depth), n_actions=6,
                                             output_activation=None,
                                             opt=keras.optimizers.Adam(learning_rate=0.0001), loss='mse'),
                'gamma': 0.99,
                'learning_rate': 0.0001,
                'frame_depth': self.frame_depth,
                'final_reward': None,
                # Use eps_initial > 1 here so only random actions used for first steps, which will make filling the
                # replay buffer more efficient. It'll also avoid decaying eps while not training.
                #'eps': EpsilonGreedy(eps_initial=1.2, decay=0.000025, eps_min=0.01, decay_schedule='compound'),
                'eps': EpsilonGreedy(eps_initial=1.1, decay=0.00001, eps_min=0.01, decay_schedule='linear'),
                'replay_buffer': ContinuousBuffer(buffer_size=10000),
                'replay_buffer_samples': 32,
                'training_history': TrainingHistory(plotting_on=self.plot_during_training,
                                                    plot_every=10, rolling_average=20,
                                                    agent_name=name)}

    def _build_for_random(self):
        name = 'RandomAgent'
        return {'name': name,
                'env_spec': self.env_spec,
                'training_history': TrainingHistory(plotting_on=self.plot_during_training,
                                                    plot_every=50, rolling_average=10,
                                                    agent_name=name)}
