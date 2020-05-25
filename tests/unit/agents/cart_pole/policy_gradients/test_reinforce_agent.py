import copy
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_almost_equal

from agents.cart_pole.policy_gradient.reinforce_agent import ReinforceAgent


class TestReinforceAgent(unittest.TestCase):
    _sut = ReinforceAgent

    @classmethod
    def setUpClass(cls) -> None:
        ReinforceAgent.set_tf(256)

    def test_env_set_during_init(self):
        # Act
        agent = self._sut()

        # Assert
        self.assertIsNotNone(agent._env)

    def test_model_set_during_init(self):
        # Act
        agent = self._sut()

        # Assert
        self.assertIsNotNone(agent._model)

    def test_history_set_during_init(self):
        # Act
        agent = self._sut()

        # Assert
        self.assertIsNotNone(agent.history)

    def test_unready_detaches_env_and_models(self):
        # Arrange
        agent = self._sut()

        # Act
        agent.unready()

        # Assert
        self.assertIsNone(agent._env)
        self.assertIsNone(agent._model)
        self.assertIsNotNone(agent._model_weights)

    def test_ready_restores_matching_object(self):
        # Arrange
        agent = self._sut()
        model_weights_original = copy.deepcopy(agent._model.get_weights())
        agent.unready()

        # Act
        agent.check_ready()

        # Assert
        self.assertIsNotNone(agent._env)
        self.assertIsNotNone(agent._model)

        model_weights = agent._model.get_weights()
        for w in range(len(model_weights)):
            assert_array_almost_equal(model_weights[w], model_weights_original[w])

    def test_play_episode_steps_returns_reward_when_not_training(self):
        # Arrange
        agent = self._sut()
        model_weights_before = copy.deepcopy(agent._model.get_weights())

        # Act
        reward = agent.play_episode(max_episode_steps=3, training=False, render=False)

        # Assert
        self.assertIsInstance(reward, float)
        model_weights = agent._model.get_weights()
        for w in range(len(model_weights_before)):
            assert_array_almost_equal(model_weights[w], model_weights_before[w])

    def test_play_episode_steps_does_not_call_update_models_when_not_training(self):
        # Arrange
        agent = self._sut()

        # Act
        with patch.object(agent, 'update_model') as mocked_update_model:
            _ = agent.play_episode(max_episode_steps=3, training=False, render=False)

        # Assert
        self.assertEqual(0, mocked_update_model.call_count)
        self.assertEqual(3, agent._env._max_episode_steps)

    def test_play_episode_steps_returns_reward_and_updates_model_when_training(self):
        # Arrange
        agent = self._sut()
        model_weights_before = copy.deepcopy(agent._model.get_weights())

        # Act
        reward = agent.play_episode(max_episode_steps=6, training=True, render=False)

        # Assert
        self.assertIsInstance(reward, float)

        model_weights = agent._model.get_weights()
        for w in range(len(model_weights_before)):
            self.assertFalse(np.all(model_weights[w].round(6) == model_weights_before[w].round(6)))

    def test_play_episode_steps_calls_update_models_when_training(self):
        # Arrange
        agent = self._sut()

        # Act
        with patch.object(agent, 'update_model') as mocked_update_model:
            _ = agent.play_episode(max_episode_steps=4, training=True, render=False)

        # Assert
        self.assertEqual(1, mocked_update_model.call_count)

    def test_train_runs_multiple_episodes(self):
        # Arrange
        agent = self._sut()

        # Act
        with patch.object(agent, 'play_episode') as mocked_play_episode:
            _ = agent.train(n_episodes=3, max_episode_steps=3, render=False)

        # Assert
        self.assertEqual(3, len(agent.history.history))
        self.assertEqual(3, mocked_play_episode.call_count)
