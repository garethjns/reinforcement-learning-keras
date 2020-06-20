from functools import partial
from typing import Any, Dict

from agents.components.history.training_history import TrainingHistory
from agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from enviroments.config_base import ConfigBase
from enviroments.doom.environment_processing.image_process_wrapper import ImageProcessWrapper
from enviroments.doom.models.conv_nn import ConvNN
from enviroments.pong.environment_processing.frame_buffer_wrapper import FrameBufferWrapper

# Wrappers as used by models

DOOM_WRAPPERS_STACK = [ImageProcessWrapper, partial(FrameBufferWrapper, obs_shape=(128, 96),
                                                    buffer_function='stack')]
DOOM_WRAPPERS_DIFF = [ImageProcessWrapper, partial(FrameBufferWrapper, obs_shape=(128, 96),
                                                   buffer_length=2,
                                                   buffer_function='diff')]


class DoomConfig(ConfigBase):
    """Defines configs for Doom."""
    _wrappers_stack = [ImageProcessWrapper, FrameBufferWrapper]
    _wrappers_diff = [ImageProcessWrapper, FrameBufferWrapper]
    env_spec = 'VizdoomBasic-v0'
    supported_agents = ('dqn', 'random')
    supported_modes = ('diff', 'stack')
    gpu_memory: int = 4096

    def __init__(self, mode: str = 'diff', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if mode not in self.supported_modes:
            raise ValueError(f"Mode {mode} is not a supported mode ({self.supported_modes})")

        if mode == "diff":
            self.env_wrappers = DOOM_WRAPPERS_DIFF
            self.frame_depth = 1
        if mode == "stack":
            self.env_wrappers = DOOM_WRAPPERS_STACK
            self.frame_depth = 3

    def build(self) -> Dict[str, Any]:

        if self.agent_type.lower() == 'dqn':
            return self._build_for_dqn()

        if self.agent_type.lower() == 'random':
            return self._build_for_random()

    def _build_for_dqn(self) -> Dict[str, Any]:

        name = 'DeepQAgent'

        return {'name': name,
                'env_spec': self.env_spec,
                'env_wrappers': self.env_wrappers,
                'model_architecture': ConvNN(observation_shape=(128, 96, self.frame_depth), n_actions=3,
                                             output_activation=None, opt='adam', learning_rate=0.0001),
                'gamma': 0.99,
                'final_reward': None,
                # Use eps_initial > 1 here so only random actions used for first steps, which will make filling the
                # replay buffer more efficient. It'll also avoid decaying eps while not training.
                # 'eps': EpsilonGreedy(eps_initial=1.2, decay=0.000025, eps_min=0.01, decay_schedule='compound'),
                'eps': EpsilonGreedy(eps_initial=1.1, decay=0.0001, eps_min=0.01, decay_schedule='linear'),
                'replay_buffer': ContinuousBuffer(buffer_size=2000),
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
