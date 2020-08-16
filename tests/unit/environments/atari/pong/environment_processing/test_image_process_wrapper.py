import unittest

from reinforcement_learning_keras.enviroments.atari.environment_processing.image_process_wrapper import \
    ImageProcessWrapper
from tests.unit.environments.atari.pong.environment_processing.env_fixture import EnvFixture


class TestImageProcessWrapper(unittest.TestCase):
    _sut = ImageProcessWrapper
    _env_fixture = EnvFixture

    def setUp(self):
        self._env = self._env_fixture(obs_shape=(250, 160, 3))

    def test_image_resize_from_250_160_3_to_expected_shape_on_reset(self):
        # Arrange
        env = self._sut(self._env)

        # Act
        obs = env.reset()

        # Assert
        self.assertEqual((84, 84), obs.shape)
        self.assertLess(obs[0, 0], 1)

    def test_image_resize_from_250_160_3_to_expected_shape_on_step(self):
        # Arrange
        env = self._sut(self._env)

        # Act
        obs, reward, done, _ = env.step(0)

        # Assert
        self.assertEqual((84, 84), obs.shape)
        self.assertLess(obs[0, 0], 1)

    def test_raises_error_with_incompatible_res(self):
        env = self._sut(self._env_fixture(obs_shape=(100, 100, 3)))

        # Act
        self.assertRaises(ValueError, lambda: env.step(0))
