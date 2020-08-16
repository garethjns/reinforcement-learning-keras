import unittest
from unittest.mock import call, Mock

from reinforcement_learning_keras.enviroments.atari.environment_processing.fire_start_wrapper import FireStartWrapper
from tests.unit.environments.atari.pong.environment_processing.env_fixture import EnvFixture


class TestFireStartWrapper(unittest.TestCase):
    _sut = FireStartWrapper
    _env_fixture = EnvFixture

    def setUp(self):
        self._env = self._env_fixture()

    def test_fire_is_first_action(self):
        # Arrange
        env = self._sut(self._env, fire_action_id=-10)
        env.env.action_indicator = Mock()

        # Act
        _ = env.reset()
        obs, reward, done, _ = env.step(action=0)

        # Assert
        env.env.action_indicator.assert_has_calls(calls=[call(-10), call(0)])

    def test_reset_returns_obs_of_expected_shape(self):
        # Arrange
        env = self._sut(self._env, fire_action_id=-10)

        # Act
        obs = env.reset()

        self.assertEqual(self._env.obs_shape, obs.shape)

    def test_step_returns_obs_of_expected_shape(self):
        # Arrange
        env = self._sut(self._env, fire_action_id=-10)
        _ = env.reset()

        # Act
        obs, reward, done, _ = env.step(action=0)

        self.assertEqual(self._env.obs_shape, obs.shape)
