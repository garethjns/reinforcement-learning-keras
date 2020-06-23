from functools import partial
from typing import Any, Dict

import gym

from agents.components.history.training_history import TrainingHistory
from agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from enviroments.atari.environment_processing.fire_start_wrapper import FireStartWrapper
from enviroments.atari.environment_processing.frame_buffer_wrapper import FrameBufferWrapper
from enviroments.atari.environment_processing.image_process_wrapper import ImageProcessWrapper
from enviroments.atari.environment_processing.max_and_skip_wrapper import MaxAndSkipWrapper
from enviroments.atari.pong.models.conv_nn import ConvNN
from enviroments.atari.pong.models.dueling_conv_nn import DuelingConvNN
from enviroments.config_base import ConfigBase

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
    supported_agents = ('dqn', 'double_dqn', 'dueling_dqn', 'double_dueling_dqn', 'random')
    supported_modes = ('diff', 'stack')
    gpu_memory: int = 4096

    @classmethod
    def env(cls):
        pass

    def unwrapped_env(self):
        pass

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

    def _build_for_dqn(self) -> Dict[str, Any]:
        return {'name': 'DeepQAgent',
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
        config_dict.update({'name': 'DuelingDQN',
                            'model_architecture': DuelingConvNN(observation_shape=(84, 84, self.frame_depth),
                                                                n_actions=6, opt='adam', learning_rate=0.000102)})

        return config_dict

    def _build_for_double_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': 'DoubleDQN',
                            'double': True,
                            'model_architecture': ConvNN(observation_shape=(84, 84, self.frame_depth),
                                                         n_actions=6, opt='adam', learning_rate=0.000102)})

        return config_dict

    def _build_for_double_dueling_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn()
        config_dict.update({'name': 'DoubleDuelingDQN',
                            'double': True,
                            'model_architecture': DuelingConvNN(observation_shape=(84, 84, self.frame_depth),
                                                                n_actions=6, opt='adam', learning_rate=0.000102)})

        return config_dict

    def _build_for_random(self):
        return {'name': 'RandomAgent',
                'env_spec': self.env_spec}
