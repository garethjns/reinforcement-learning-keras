import unittest

import numpy as np

from reinforcement_learning_keras.environments.gfootball.environment_processing.simple_and_smm_obs_wrapper import \
    SimpleAndSMMObsWrapper
from reinforcement_learning_keras.environments.gfootball.environment_processing.smm_frame_process_wrapper import \
    SMMFrameProcessWrapper
from tests.unit.environments.atari.pong.environment_processing.env_fixture import EnvFixture

try:
    from gfootball.env.config import Config
    from gfootball.env.football_env import FootballEnv

    GFOOTBALL_AVAILABLE = True
except ImportError:
    GFOOTBALL_AVAILABLE = False


class TestSMMFrameProcessWrapper(unittest.TestCase):

    def setUp(self):
        self._obs_shape = (72, 96, 4)
        self._env_fixture = EnvFixture(obs_shape=self._obs_shape)
        self._env_fixture._obs = np.zeros(self._obs_shape) + 255
        self._sut = SMMFrameProcessWrapper(self._env_fixture)

        frames = [np.random.randint(low=0, high=255, size=(5, 5, 1)) + i for i in range(4)]
        self._random_obs = np.concatenate(frames, axis=2)

    def test_normalise_single_frame(self):
        # Act
        norm_frame = self._sut._normalise_frame(self._random_obs)

        # Assert
        self.assertEqual(norm_frame.shape, self._random_obs.shape)
        self.assertGreaterEqual(norm_frame.min(), 0)
        self.assertLessEqual(norm_frame.max(), 1)

    def test_normalise_all_frames(self):
        # Act
        norm_frame = self._sut._normalise_frame(self._random_obs[..., 0])

        # Assert
        self.assertEqual(norm_frame.shape, self._random_obs.shape[0: 2])
        self.assertGreaterEqual(norm_frame.min(), 0)
        self.assertLessEqual(norm_frame.max(), 1)

    def test_build_buffer_of_2_obs(self):
        # Act
        obs, _, _, _ = self._sut.step(3)
        obs2, _, _, _ = self._sut.step(3)

        # Assert
        # First obs has difference of 1 (255) compared to initial frame buffer of 0
        self.assertEqual(self._obs_shape, obs.shape)
        self.assertEqual(1, np.unique(obs))
        # Second obs should be all 0 due to 1 (255) - 1 (255)
        self.assertEqual(self._obs_shape, obs.shape)
        self.assertEqual(0, np.unique(obs2))

    def test_process_with_array_input(self):
        # Arrange
        obs = np.ones((4, 4, 2))
        obs2 = obs + 2
        buffer = SMMFrameProcessWrapper(obs_shape=obs.shape)
        initial_buffer = buffer._obs_buffer[0]

        # Act
        first_call = buffer.process(obs)
        second_call = buffer.process(obs2)

        # Assert
        self.assertIsInstance(first_call, np.ndarray)
        self.assertIsInstance(second_call, np.ndarray)
        self.assertEqual(obs.shape, first_call.shape)
        self.assertEqual(obs.shape, second_call.shape)
        self.assertEqual(0, np.unique(initial_buffer))
        self.assertNotEqual(0, np.unique(buffer._obs_buffer[0]))
        self.assertAlmostEqual(1 / 255.0, float(np.unique(first_call)))
        self.assertEqual(2 / 255.0, float(np.unique(second_call)))

    def test_process_with_tuple_input(self):
        # Arrange
        obs = (np.ones((4, 4, 2)), ['other', 'obs'])
        obs2 = (obs[0] + 2, ['other', 'obs'])
        buffer = SMMFrameProcessWrapper(obs_shape=obs[0].shape)
        initial_buffer = buffer._obs_buffer[0]

        # Act
        first_call = buffer.process(obs)
        second_call = buffer.process(obs2)

        # Assert
        self.assertIsInstance(first_call, tuple)
        self.assertIsInstance(second_call, tuple)
        self.assertEqual(obs[0].shape, first_call[0].shape)
        self.assertEqual(obs[0].shape, second_call[0].shape)
        self.assertEqual(0, np.unique(initial_buffer))
        self.assertNotEqual(0, np.unique(buffer._obs_buffer[0]))
        self.assertAlmostEqual(1 / 255.0, float(np.unique(first_call[0])))
        self.assertEqual(first_call[1:], (['other', 'obs'],))
        self.assertEqual(2 / 255.0, float(np.unique(second_call[0])))
        self.assertEqual(second_call[1:], (['other', 'obs'],))

    @unittest.skipUnless(GFOOTBALL_AVAILABLE, "GFootball not available in this env")
    def test_stacked_with_simple_and_smm_obs_wrapper_on_reset(self):
        # Arrange
        env = SMMFrameProcessWrapper(SimpleAndSMMObsWrapper(FootballEnv(config=Config())))

        # Act
        obs = env.reset()

        # Assert
        self.assertEqual(1, np.max(obs[0]))
        self.assertEqual(0, np.min(obs[0]))
        self.assertEqual((72, 96, 4), obs[0].shape)
        self.assertEqual((115,), obs[1].shape)

    @unittest.skipUnless(GFOOTBALL_AVAILABLE, "GFootball not available in this env")
    def test_stacked_with_simple_and_smm_obs_wrapper_on_step(self):
        # Arrange
        env = SMMFrameProcessWrapper(SimpleAndSMMObsWrapper(FootballEnv(config=Config())))
        obs = env.reset()

        # Act
        first_call, _, _, _ = env.step(0)
        second_call, _, _, _ = env.step(0)

        # Assert
        self.assertEqual(1, np.max(first_call[0]))
        self.assertEqual(-1, np.min(first_call[0]))
        self.assertEqual(1, np.max(second_call[0]))
        self.assertEqual(-1, np.min(second_call[0]))
        self.assertEqual((72, 96, 4), obs[0].shape)
        self.assertEqual((115,), obs[1].shape)
