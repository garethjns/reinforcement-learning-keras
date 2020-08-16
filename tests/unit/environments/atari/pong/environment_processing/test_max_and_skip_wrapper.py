import unittest
from unittest.mock import call, Mock

import numpy as np

from reinforcement_learning_keras.enviroments.atari.environment_processing.max_and_skip_wrapper import MaxAndSkipWrapper
from tests.unit.environments.atari.pong.environment_processing.env_fixture import EnvFixture


class TestMaxAndSkipWrapper(unittest.TestCase):
    _sut = MaxAndSkipWrapper
    _env_fixture = EnvFixture

    def setUp(self):
        self._env = self._env_fixture()

    def test_max_aggregation_of_state_returns_expected_step_outputs(self):
        # Arrange
        frame_buffer_length = 3
        n_action_frames = 6
        env = self._sut(self._env,
                        frame_buffer_length=frame_buffer_length, n_action_frames=n_action_frames)
        env.env.action_indicator = Mock()

        # Act
        _ = env.reset()
        obs, reward, done, _ = env.step(action=0)

        # Assert
        self.assertEqual(1, np.unique(obs))
        self.assertAlmostEqual(n_action_frames, reward)
        self.assertEqual(False, done)
        env.env.action_indicator.assert_has_calls(calls=[call(0) for _ in range(n_action_frames)])
        self.assertEqual(frame_buffer_length, len(env._frame_buffer))

    def test_sum_aggregation_of_state_returns_expected_step_outputs(self):
        # Arrange
        frame_buffer_length = 6
        n_action_frames = 8
        env = self._sut(self._env, frame_buffer_agg_f=np.sum,
                        frame_buffer_length=frame_buffer_length, n_action_frames=n_action_frames)
        env.env.action_indicator = Mock()

        # Act
        _ = env.reset()
        obs, reward, done, _ = env.step(action=-1)

        # Assert
        self.assertEqual(frame_buffer_length, np.unique(obs))
        self.assertAlmostEqual(n_action_frames, reward)
        self.assertEqual(False, done)
        env.env.action_indicator.assert_has_calls(calls=[call(-1) for _ in range(n_action_frames)])
        self.assertEqual(frame_buffer_length, len(env._frame_buffer))

    def test_aggregation_of_state_returns_expected_step_outputs_with_done_in_sequence(self):
        # Arrange
        frame_buffer_length = 5
        n_action_frames = 6
        env = self._sut(self._env, frame_buffer_length=frame_buffer_length,
                        n_action_frames=n_action_frames)
        env.env.action_indicator = Mock()

        # Act
        # Action 1 will cause fixture to return done after 2 turns. Done turn has additional reward.
        _ = env.reset()
        obs, reward, done, _ = env.step(action=1)

        # Assert
        self.assertEqual(2, np.unique(obs))
        self.assertAlmostEqual(4.0, reward)  # 3 steps +1 bonus reward
        self.assertEqual(True, done)
        env.env.action_indicator.assert_has_calls(calls=[call(1) for _ in range(3)])
        self.assertEqual(4, len(env._frame_buffer))  # 3 + 1 from the rest call, not 5
