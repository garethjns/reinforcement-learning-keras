import os
from functools import partial
from typing import Dict, Any

from reinforcement_learning_keras.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from reinforcement_learning_keras.agents.models.conv_nn import ConvNN
from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from reinforcement_learning_keras.enviroments.atari.atari_default_config import AtariDefaultConfig
from reinforcement_learning_keras.enviroments.atari.environment_processing.fire_start_wrapper import FireStartWrapper
from reinforcement_learning_keras.enviroments.atari.environment_processing.frame_buffer_wrapper import \
    FrameBufferWrapper
from reinforcement_learning_keras.enviroments.atari.environment_processing.image_process_wrapper import \
    ImageProcessWrapper
from reinforcement_learning_keras.enviroments.atari.environment_processing.max_and_skip_wrapper import MaxAndSkipWrapper


class SpaceInvadersConfig(AtariDefaultConfig):
    """Defines configs tweaks for Space Invaders."""
    env_spec = 'SpaceInvadersNoFrameskip-v0'

    _wrappers_stack = (partial(MaxAndSkipWrapper, frame_buffer_length=4),
                       ImageProcessWrapper,
                       FireStartWrapper,
                       FrameBufferWrapper)
    _wrappers_diff = (partial(MaxAndSkipWrapper, frame_buffer_length=4),
                      ImageProcessWrapper,
                      FireStartWrapper,
                      partial(FrameBufferWrapper, buffer_length=2, buffer_function='diff'))

    def _build_for_dqn(self) -> Dict[str, Any]:
        return {'name': os.path.join(self.folder, 'DeepQAgent'),
                'env_spec': self.env_spec,
                'env_wrappers': self.env_wrappers,
                'model_architecture': ConvNN(observation_shape=(84, 84, self.frame_depth), n_actions=6,
                                             output_activation=None, opt='adam', learning_rate=0.00008),
                'gamma': 0.99,
                'final_reward': None,
                'eps': EpsilonGreedy(eps_initial=2, decay=0.000025, eps_min=0.01, decay_schedule='linear'),
                'replay_buffer': ContinuousBuffer(buffer_size=40000),
                'replay_buffer_samples': 32}

