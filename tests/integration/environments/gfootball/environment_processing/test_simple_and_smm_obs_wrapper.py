import unittest
from typing import Tuple

import numpy as np

from reinforcement_learning_keras.environments.gfootball.environment_processing.simple_and_smm_obs_wrapper import \
    SimpleAndSMMObsWrapper

try:
    from gfootball.env.config import Config
    from gfootball.env.football_env import FootballEnv

    GFOOTBALL_AVAILABLE = True
except ImportError:
    GFOOTBALL_AVAILABLE = False


@unittest.skipUnless(GFOOTBALL_AVAILABLE, "GFootball not available in this env.")
class TestSimpleAndSMMObsWrapper(unittest.TestCase):
    def setUp(self):
        self._env = FootballEnv(config=Config())
        self._sut = SimpleAndSMMObsWrapper

    def _assert_obs_shape(self, obs: Tuple[np.ndarray, np.ndarray]):
        self.assertEqual(2, len(obs))
        self.assertEqual((1, 72, 96, 4), obs[0].shape)
        self.assertEqual((115,), obs[1].shape)

    def test_shapes_as_expected_with_env_on_reset(self):
        # Arrange
        wrapped_env = self._sut(self._env)

        # Act
        obs = wrapped_env.reset()

        # Assert
        self._assert_obs_shape(obs)

    def test_shapes_as_expected_without_env_on_reset(self):
        # Arrange
        obs = self._env.reset()

        # Act
        obs = self._sut.process_obs(obs)

        # Assert
        self._assert_obs_shape(obs)

    def test_shapes_as_expected_with_env_on_step(self):
        # Arrange
        wrapped_env = self._sut(self._env)
        wrapped_env.reset()

        # Act
        obs, reward, done, info = wrapped_env.step(0)

        # Assert
        self._assert_obs_shape(obs)

    def test_shapes_as_expected_without_env_on_step(self):
        # Arrange
        obs = self._env.reset()
        obs, reward, done, info = self._env.step(0)

        # Act
        obs = self._sut.process_obs(obs)

        # Assert
        self._assert_obs_shape(obs)
