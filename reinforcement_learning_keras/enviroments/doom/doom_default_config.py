import os
from functools import partial
from typing import Any, Dict

# Although unused, this import will register Doom envs with Gym
# noinspection PyUnresolvedReferences
import vizdoomgym

from reinforcement_learning_keras.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from reinforcement_learning_keras.enviroments.config_base import ConfigBase
from reinforcement_learning_keras.enviroments.doom.environment_processing.frame_buffer_wrapper import FrameBufferWrapper
from reinforcement_learning_keras.enviroments.doom.environment_processing.image_process_wrapper import \
    ImageProcessWrapper
from reinforcement_learning_keras.enviroments.doom.models.conv_nn import ConvNN


class DoomDefaultConfig(ConfigBase):
    """Defines default configs for Doom."""
    env_spec = 'VizdoomBasic-v0'
    n_actions = 3
    supported_agents = ('dqn', 'random')
    supported_modes = ('diff', 'stack')
    gpu_memory: int = 2048

    _wrappers_stack = (ImageProcessWrapper, partial(FrameBufferWrapper, obs_shape=(96, 128),
                                                    buffer_function='stack'))
    _wrappers_diff = (ImageProcessWrapper, partial(FrameBufferWrapper, obs_shape=(96, 128),
                                                   buffer_length=2,
                                                   buffer_function='diff'))

    def __init__(self, mode: str = 'diff', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if mode not in self.supported_modes:
            raise ValueError(f"Mode {mode} is not a supported mode ({self.supported_modes})")

        self.mode = mode
        if self.mode == "diff":
            self.env_wrappers = self._wrappers_diff
            self.frame_depth = 1
            self.wrapped_env = FrameBufferWrapper(ImageProcessWrapper(self.unwrapped_env),
                                                  obs_shape=(96, 128), buffer_length=2, buffer_function='diff')
        if self.mode == "stack":
            self.env_wrappers = self._wrappers_stack
            self.frame_depth = 3
            self.wrapped_env = FrameBufferWrapper(ImageProcessWrapper(self.unwrapped_env),
                                                  obs_shape=(96, 128), buffer_function='stack')

    def _build_for_dqn(self) -> Dict[str, Any]:
        return {'name': os.path.join(self.folder, 'DeepQAgent'),
                'env_spec': self.env_spec,
                'env_wrappers': self.env_wrappers,
                'model_architecture': ConvNN(observation_shape=(96, 128, self.frame_depth), n_actions=self.n_actions,
                                             output_activation=None, opt='adam', learning_rate=0.0001),
                'gamma': 0.99,
                'final_reward': None,
                'eps': EpsilonGreedy(eps_initial=1.2, decay=0.00002, eps_min=0.01, decay_schedule='linear'),
                'replay_buffer': ContinuousBuffer(buffer_size=10000),
                'replay_buffer_samples': 32}

    def _build_for_random(self):
        return {'name': os.path.join(self.folder, 'RandomAgent'),
                'env_spec': self.env_spec}
