import os
from functools import partial
from typing import Any, Dict

from reinforcement_learning_keras.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from reinforcement_learning_keras.enviroments.atari.environment_processing.fire_start_wrapper import FireStartWrapper
from reinforcement_learning_keras.enviroments.atari.environment_processing.frame_buffer_wrapper import \
    FrameBufferWrapper
from reinforcement_learning_keras.enviroments.atari.environment_processing.image_process_wrapper import \
    ImageProcessWrapper
from reinforcement_learning_keras.enviroments.atari.environment_processing.max_and_skip_wrapper import MaxAndSkipWrapper
from reinforcement_learning_keras.agents.models.conv_nn import ConvNN
from reinforcement_learning_keras.agents.models.dueling_conv_nn import DuelingConvNN
from reinforcement_learning_keras.enviroments.config_base import ConfigBase


class AtariDefaultConfig(ConfigBase):
    """Defines configs for Pong."""
    env_spec: str
    supported_agents = ('dqn', 'double_dqn', 'dueling_dqn', 'double_dueling_dqn', 'random')
    supported_modes = ('diff', 'stack')
    gpu_memory: int = 2048

    _wrappers_stack = (MaxAndSkipWrapper, ImageProcessWrapper, FireStartWrapper, FrameBufferWrapper)
    _wrappers_diff = (MaxAndSkipWrapper, ImageProcessWrapper, FireStartWrapper,
                      partial(FrameBufferWrapper, buffer_length=2, buffer_function='diff'))

    def __init__(self, *args, mode: str = 'diff', **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if mode not in self.supported_modes:
            raise ValueError(f"Mode {mode} is not a supported mode ({self.supported_modes})")

        self.mode = mode
        if self.mode == "diff":
            self.env_wrappers = self._wrappers_diff
            self.frame_depth = 1
            self.wrapped_env = FrameBufferWrapper(
                FireStartWrapper(ImageProcessWrapper(MaxAndSkipWrapper(self.unwrapped_env))),
                buffer_length=2, buffer_function='diff')
        if self.mode == "stack":
            self.env_wrappers = self._wrappers_stack
            self.frame_depth = 3
            self.wrapped_env = FrameBufferWrapper(FireStartWrapper(
                ImageProcessWrapper(MaxAndSkipWrapper(self.unwrapped_env))),
                buffer_length=3, buffer_function='stack')

    def _build_for_dqn(self) -> Dict[str, Any]:
        return {'name': os.path.join(self.folder, 'DeepQAgent'),
                'env_spec': self.env_spec,
                'env_wrappers': self.env_wrappers,
                'model_architecture': ConvNN(observation_shape=(84, 84, self.frame_depth), n_actions=6,
                                             output_activation=None, opt='adam', learning_rate=0.000105),
                'gamma': 0.99,
                'final_reward': None,
                # Use eps_initial > 1 here so only random actions used for first steps, which will make filling the
                # replay buffer more efficient. It'll also avoid decaying eps while not training.
                # Alternative: 'eps': EpsilonGreedy(eps_initial=1.2, decay=0.000025, eps_min=0.01,
                #                                   decay_schedule='compound'),
                'eps': EpsilonGreedy(eps_initial=1.1, decay=0.00001, eps_min=0.01, decay_schedule='linear'),
                'replay_buffer': ContinuousBuffer(buffer_size=10000),
                'replay_buffer_samples': 32}

    def _build_for_dueling_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': os.path.join(self.folder, 'DuelingDQN'),
                            'model_architecture': DuelingConvNN(observation_shape=(84, 84, self.frame_depth),
                                                                n_actions=6, opt='adam', learning_rate=0.000102)})

        return config_dict

    def _build_for_double_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': os.path.join(self.folder, 'DoubleDQN'),
                            'double': True,
                            'model_architecture': ConvNN(observation_shape=(84, 84, self.frame_depth),
                                                         n_actions=6, opt='adam', learning_rate=0.000102)})

        return config_dict

    def _build_for_double_dueling_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': os.path.join(self.folder, 'DoubleDuelingDQN'),
                            'double': True,
                            'model_architecture': DuelingConvNN(observation_shape=(84, 84, self.frame_depth),
                                                                n_actions=6, opt='adam', learning_rate=0.000102)})

        return config_dict

    def _build_for_random(self):
        return {'name': os.path.join(self.folder, 'RandomAgent'),
                'env_spec': self.env_spec}
