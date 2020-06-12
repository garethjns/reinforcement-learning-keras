from functools import partial

import gym

from agents.pong.environment_processing.fire_start_wrapper import FireStartWrapper
from agents.pong.environment_processing.frame_buffer_wrapper import FrameBufferWrapper
from agents.pong.environment_processing.image_process_wrapper import ImageProcessWrapper
from agents.pong.environment_processing.max_and_skip_wrapper import MaxAndSkipWrapper

# Wrappers as used by models
PONG_WRAPPERS_STACK = [MaxAndSkipWrapper, ImageProcessWrapper, FireStartWrapper, FrameBufferWrapper]
PONG_WRAPPERS_DIFF = [MaxAndSkipWrapper, ImageProcessWrapper, FireStartWrapper,
                      partial(FrameBufferWrapper,
                              buffer_length=2,
                              buffer_function='diff')]

# Envs as used by models
PONG_ENV_SPEC = "PongNoFrameskip-v4"
PONG_ENV = gym.make("PongNoFrameskip-v4")
PONG_ENV_STACK = FrameBufferWrapper(FireStartWrapper(ImageProcessWrapper(MaxAndSkipWrapper(PONG_ENV))))
PONG_ENV_DIFF = FrameBufferWrapper(FireStartWrapper(ImageProcessWrapper(MaxAndSkipWrapper(PONG_ENV))),
                                   buffer_length=2,
                                   buffer_function='diff')
