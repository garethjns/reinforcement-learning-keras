import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from reinforcement_learning_keras.enviroments.atari.environment_processing.frame_buffer_wrapper import \
    FrameBufferWrapper
from tests.unit.environments.atari.pong.environment_processing.env_fixture import EnvFixture


class TestFrameBufferWrapper(unittest.TestCase):
    _sut = FrameBufferWrapper
    _env_fixture = EnvFixture

    def setUp(self):
        self._env = self._env_fixture()

    def test_reset_returns_expected_obs_values_and_shapes_with_stack_op(self):
        # Arrange
        env = self._sut(self._env, obs_shape=self._env.obs_shape)

        # Act
        obs = env.reset()

        # Assert
        self.assertEqual((self._env.obs_shape[0], self._env.obs_shape[1], 3),
                         obs.shape)
        self.assertEqual(np.unique(obs[:, :, 0:2]), 0)

    def test_step_returns_expected_obs_values_and_shapes_with_stack_op(self):
        # Arrange
        env = self._sut(self._env, obs_shape=self._env.obs_shape)

        # Act
        obs1 = env.reset()
        obs2, reward, done, _ = env.step(0)

        # Assert
        self.assertEqual((self._env.obs_shape[0], self._env.obs_shape[1], 3),
                         obs2.shape)
        self.assertEqual(np.unique(obs2[:, :, 0]), 0)
        assert_array_almost_equal(obs1[:, :, 2], obs2[:, :, 1])
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_reset_returns_expected_obs_values_and_shapes_with_diff_op(self):
        # Arrange
        env = self._sut(self._env, obs_shape=self._env.obs_shape,
                        buffer_length=2, buffer_function='diff')

        # Act
        obs = env.reset()

        # Assert
        self.assertEqual((self._env.obs_shape[0], self._env.obs_shape[1], 1),
                         obs.shape)
        self.assertEqual(np.unique(obs[:, :]), 1)

    def test_step_returns_expected_obs_values_and_shapes_with_diff_op(self):
        # Arrange
        env = self._sut(self._env, obs_shape=self._env.obs_shape,
                        buffer_length=2, buffer_function='diff')

        # Act
        _ = env.reset()
        obs2, reward, done, _ = env.step(0)
        # Assert
        self.assertEqual((self._env.obs_shape[0], self._env.obs_shape[1], 1),
                         obs2.shape)
        self.assertEqual(0, np.unique(obs2[:, :]))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
