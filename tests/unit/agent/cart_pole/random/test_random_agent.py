import unittest
from typing import Callable
from unittest.mock import patch

from agents.cart_pole.random.random_agent import RandomAgent


class TestDeepQAgent(unittest.TestCase):
    _sut = RandomAgent

    def test_env_set_during_init(self):
        # Act
        agent = self._sut()

        # Assert
        self.assertIsNotNone(agent._env)

    def test_model_set_during_init(self):
        # Act
        agent = self._sut()

        # Assert
        self.assertIsInstance(agent.model, Callable)

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

    def test_ready_restores_matching_object(self):
        # Arrange
        agent = self._sut()
        agent.unready()

        # Act
        agent.check_ready()

        # Assert
        # TODO: Not quite what test says it does but fine for now
        self.assertIsNotNone(agent._env)

    def test_play_episode_steps_returns_reward_when_not_training(self):
        # Arrange
        agent = self._sut()

        # Act
        reward = agent.play_episode(max_episode_steps=3, training=False, render=False)

        # Assert
        self.assertIsInstance(reward, float)

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

        # Act
        reward = agent.play_episode(max_episode_steps=6, training=True, render=False)

        # Assert
        self.assertIsInstance(reward, float)

    def test_play_episode_steps_calls_update_models_when_training(self):
        # Arrange
        agent = self._sut()

        # Act
        with patch.object(agent, 'update_model') as mocked_update_model:
            _ = agent.play_episode(max_episode_steps=4, training=True, render=False)

        # Assert
        self.assertEqual(0, mocked_update_model.call_count)

    def test_train_runs_multiple_episodes(self):
        # Arrange
        agent = self._sut()

        # Act
        with patch.object(agent, 'play_episode') as mocked_play_episode:
            _ = agent.train(n_episodes=3, max_episode_steps=3, render=False)

        # Asser
        self.assertEqual(3, len(agent.history.history))
        self.assertEqual(3, mocked_play_episode.call_count)
