import os
from abc import ABC
from functools import partial
from typing import Any, Dict, Tuple

# Although unused, this import will register Doom envs with Gym
# noinspection PyUnresolvedReferences
import vizdoomgym

from reinforcement_learning_keras.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from reinforcement_learning_keras.agents.models.conv_nn import ConvNN
from reinforcement_learning_keras.agents.models.dueling_conv_nn import DuelingConvNN
from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from reinforcement_learning_keras.enviroments.config_base import ConfigBase
from reinforcement_learning_keras.enviroments.doom.environment_processing.frame_buffer_wrapper import FrameBufferWrapper
from reinforcement_learning_keras.enviroments.doom.environment_processing.image_process_wrapper import \
    ImageProcessWrapper


class DoomDefaultConfig(ConfigBase, ABC):
    """Defines default configs for Doom."""
    env_spec = 'VizdoomBasic-v0'
    n_actions = 3
    supported_agents = ('dqn', 'dueling_dqn', 'double_dqn', 'double_dueling_dqn', 'random')
    supported_modes = ('diff', 'stack')
    gpu_memory: int = 2048

    # 3 different possible resolutions:
    # (96, 128) @ 40%
    # (96, 128) @ 20%
    # (90, 160) @ 20%
    res_scale: float = 0.4
    target_obs_shape: Tuple[int, int] = (96, 128)

    def __init__(self, *args, mode: str = 'diff', **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if mode not in self.supported_modes:
            raise ValueError(f"Mode {mode} is not a supported mode ({self.supported_modes})")

        self._wrappers_stack = (partial(ImageProcessWrapper, scale=self.res_scale),
                                partial(FrameBufferWrapper, obs_shape=self.target_obs_shape, buffer_function='stack'))

        self._wrappers_diff = (partial(ImageProcessWrapper, scale=self.res_scale),
                               partial(FrameBufferWrapper, obs_shape=self.target_obs_shape, buffer_length=2,
                                       buffer_function='diff'))

        self.mode = mode
        if self.mode == "diff":
            self.env_wrappers = self._wrappers_diff
            self.frame_depth = 1

        if self.mode == "stack":
            self.env_wrappers = self._wrappers_stack
            self.frame_depth = 3

        self.wrapped_env = self.env_wrappers[1](self.env_wrappers[0](self.unwrapped_env))

    def _build_for_dqn(self) -> Dict[str, Any]:
        return {'name': os.path.join(self.folder, 'DeepQAgent'),
                'env_spec': self.env_spec,
                'env_wrappers': self.env_wrappers,
                'model_architecture': ConvNN(
                    observation_shape=(self.target_obs_shape[0], self.target_obs_shape[1], self.frame_depth),
                    n_actions=self.n_actions, output_activation=None, opt='adam', learning_rate=0.0001),
                'gamma': 0.99,
                'final_reward': None,
                'eps': EpsilonGreedy(eps_initial=1.2, decay=0.0000019, eps_min=0.01, decay_schedule='linear'),
                'replay_buffer': ContinuousBuffer(buffer_size=20000),
                'replay_buffer_samples': 32}

    def _build_for_dueling_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': os.path.join(self.folder, 'DuelingDQN'),
                            'model_architecture': DuelingConvNN(
                                observation_shape=(self.target_obs_shape[0], self.target_obs_shape[1],
                                                   self.frame_depth),
                                n_actions=6, opt='adam', learning_rate=0.0001)})

        return config_dict

    def _build_for_double_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': os.path.join(self.folder, 'DoubleDQN'),
                            'double': True,
                            'model_architecture': ConvNN(
                                observation_shape=(self.target_obs_shape[0], self.target_obs_shape[1],
                                                   self.frame_depth),
                                n_actions=self.n_actions, output_activation=None, opt='adam', learning_rate=0.0001)})

        return config_dict

    def _build_for_double_dueling_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': os.path.join(self.folder, 'DoubleDuelingDQN'),
                            'double': True,
                            'model_architecture': DuelingConvNN(
                                observation_shape=(self.target_obs_shape[0], self.target_obs_shape[1],
                                                   self.frame_depth),
                                n_actions=self.n_actions, opt='adam', learning_rate=0.0001)})

        return config_dict

    def _build_for_random(self):
        return {'name': os.path.join(self.folder, 'RandomAgent'),
                'env_spec': self.env_spec}
