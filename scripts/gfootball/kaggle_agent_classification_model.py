"""
This file defines a self-contained kaggle-compatible agent definition for use with kaggle_environments runner.

This is the pretrained version of the model.
"""
import collections
import os
from typing import Any, Union, List

import gym
import numpy as np
from gfootball.env.wrappers import Simple115StateWrapper
from tensorflow import keras


class BufferWrapper(gym.Wrapper):
    """General buffer wrapper to handle raw observations."""

    def __init__(self, env: Union[None, gym.Env] = None, buffer_length: int = 2) -> None:
        if env is not None:
            super().__init__(env)
        else:
            self.env = None

        self._buffer_length = buffer_length
        self._prepare_obs_buffer()

    def _prepare_obs_buffer(self) -> None:
        """Create buffer and preallocate with empty arrays of expected shape."""
        self._obs_buffer = collections.deque(maxlen=self._buffer_length)

        for _ in range(self._buffer_length):
            self._obs_buffer.append(None)

    def add(self, obs: Any):
        self._obs_buffer.append(obs)

    def get(self) -> List[Any]:
        return [self._obs_buffer[n] for n in range(self._buffer_length)]


FN = "nn_s115_pretrained_model"
KAGGLE_PATH = f"/kaggle_simulations/agent/{FN}"
if os.path.exists(KAGGLE_PATH):
    # On kaggle
    path = KAGGLE_PATH
else:
    # Local, could be in scripts/ or not
    if os.path.exists(FN):
        path = FN
    else:
        path = f"../{FN}"

mod = keras.models.load_model(path)
buffer = BufferWrapper(buffer_length=2)


def agent(obs):
    global mod
    global buffer

    buffer.add(obs)
    buffered_obs = buffer.get()

    s115_obs = []
    for b in buffered_obs:
        if b is None:
            # First step
            s115_obs.append(np.zeros(shape=(1, 115)))
        else:
            s115_obs.append(Simple115StateWrapper.convert_observation(b['players_raw'], fixed_positions=True))

    obs = np.concatenate(s115_obs, axis=1)

    action = mod.predict(obs).argmax()

    return [int(action)]
