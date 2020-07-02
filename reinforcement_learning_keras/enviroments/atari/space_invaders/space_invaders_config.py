from functools import partial

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
