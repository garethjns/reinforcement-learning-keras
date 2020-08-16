import unittest

from reinforcement_learning_keras.enviroments.doom.environment_processing.image_process_wrapper import \
    ImageProcessWrapper
from tests.unit.environments.atari.pong.environment_processing.env_fixture import EnvFixture


class TestImageProcessWrapper(unittest.TestCase):
    _sut = ImageProcessWrapper
    _env_fixture = EnvFixture

    def setUp(self):
        self._env = self._env_fixture(obs_shape=(320, 225, 3))

    def test_image_resize_40pc_to_expected_shape_on_reset(self):
        # Arrange
        env = self._sut(self._env)

        # Act
        obs = env.reset()

        # Assert
        self.assertEqual((128, 90), obs.shape)
        self.assertLess(obs[0, 0], 1)

    def test_image_resize_40pc_to_expected_shape_on_step(self):
        # Arrange
        env = self._sut(self._env)

        # Act
        obs, reward, done, _ = env.step(0)

        # Assert
        self.assertEqual((128, 90), obs.shape)
        self.assertLess(obs[0, 0], 1)
