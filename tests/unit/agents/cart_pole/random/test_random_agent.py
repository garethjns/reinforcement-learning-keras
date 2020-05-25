import unittest
from typing import Callable
from unittest.mock import patch

from agents.agent_base import AgentBase
from agents.cart_pole.random.random_agent import RandomAgent


class TestRandomAgent(unittest.TestCase):
    """
    The random agent is basically a non-abstract version of AgentBase. It's used here to define the general test
    interface, as if it's a mocked AgentBase.

    Other agents can modify the private methods here to define how to arrange/assert for their specific cases.
    Acting is standard across agents, except for where path.object is required for multiple methods (see DQN as an
    example).
    """
    _sut = RandomAgent

    def _standard_set_up(self):
        self._n_step = 4
        self._n_episodes = 3
        self._expected_model_update_during_training_episode: int = self._n_step
        self._expected_model_update_during_playing_episode: int = 0
        self._expected_play_episode_calls = self._n_episodes

    def _agent_specific_set_up(self):
        # This agent doesn't bother calling update_model as it doesn't have one.
        self._expected_model_update_during_training_episode: int = 0
        self._expected_model_update_after_training_episode: int = 0

    def setUp(self):
        self._standard_set_up()
        self._agent_specific_set_up()

    def _ready_agent(self) -> RandomAgent:
        return self._sut()

    @staticmethod
    def _checkpoint_model(agent: AgentBase) -> None:
        """No model to checkpoint."""
        return None

    def _assert_model_unchanged(self, agent: AgentBase, checkpoint: None) -> None:
        """No model to compare, nothing to assert."""
        pass

    def _assert_model_changed(self, agent: AgentBase, checkpoint: None) -> None:
        """No model to compare, nothing to assert."""
        pass

    def _assert_agent_unready(self, agent: AgentBase) -> None:
        self.assertIsNone(agent._env)

    def _assert_agent_ready(self, agent: RandomAgent) -> None:
        self.assertIsNotNone(agent._env)
        self.assertIsInstance(agent.model, Callable)

    def test_env_set_during_init(self) -> None:
        # Act
        agent = self._ready_agent()

        # Assert
        self.assertIsNotNone(agent._env)

    def test_model_set_during_init(self) -> None:
        # Act
        agent = self._sut()

        # Assert
        self._assert_agent_ready(agent)

    def test_history_set_during_init(self) -> None:
        # Act
        agent = self._ready_agent()

        # Assert
        self.assertIsNotNone(agent.history)

    def test_unready_detaches_env_and_models(self) -> None:
        # Arrange
        agent = self._ready_agent()

        # Act
        agent.unready()

        # Assert
        self._assert_agent_unready(agent)

    def test_ready_restores_matching_object(self) -> None:
        # Arrange
        agent = self._ready_agent()
        checkpoint = self._checkpoint_model(agent)
        agent.unready()

        # Act
        agent.check_ready()

        # Assert
        self._assert_agent_ready(agent)
        self._assert_model_unchanged(agent, checkpoint)

    def test_play_episode_steps_returns_reward_when_not_training(self) -> None:
        # Arrange
        agent = self._ready_agent()

        # Act
        reward = agent.play_episode(max_episode_steps=3, training=False, render=False)

        # Assert
        self.assertIsInstance(reward, float)

    def test_play_episode_steps_does_not_call_update_models_when_not_training(self):
        # Arrange
        agent = self._ready_agent()

        # Act
        with patch.object(agent, 'update_model') as mocked_update_model:
            _ = agent.play_episode(max_episode_steps=self._n_step, training=False, render=False)

        # Assert
        self.assertEqual(self._expected_model_update_during_playing_episode, mocked_update_model.call_count)
        self.assertEqual(self._n_step, agent._env._max_episode_steps)

    def test_play_episode_steps_does_not_update_models_when_not_training(self):
        # Arrange
        agent = self._ready_agent()
        checkpoint = self._checkpoint_model(agent)

        # Act
        _ = agent.play_episode(max_episode_steps=self._n_step, training=False, render=False)

        # Assert
        self._assert_model_unchanged(agent, checkpoint)

    def test_play_episode_steps_returns_reward_and_updates_model_when_training(self) -> None:
        # Arrange
        agent = self._ready_agent()

        # Act
        reward = agent.play_episode(max_episode_steps=self._n_step, training=True, render=False)

        # Assert
        self.assertIsInstance(reward, float)

    def test_play_episode_steps_calls_update_models_as_expected_when_training(self) -> None:
        # Arrange
        agent = self._ready_agent()

        # Act
        with patch.object(agent, 'update_model') as mocked_update_model:
            _ = agent.play_episode(max_episode_steps=self._n_step, training=True, render=False)

        # Assert
        self.assertEqual(self._expected_model_update_during_training_episode, mocked_update_model.call_count)

    def test_play_episode_steps_updates_models_as_expected_when_training(self) -> None:
        # Arrange
        agent = self._ready_agent()
        checkpoint = self._checkpoint_model(agent)

        # Act
        _ = agent.play_episode(max_episode_steps=self._n_step, training=True, render=False)

        # Assert
        self._assert_model_changed(agent, checkpoint)

    def test_train_runs_multiple_episodes(self) -> None:
        # Arrange
        agent = self._sut()

        # Act
        with patch.object(agent, 'play_episode') as mocked_play_episode:
            agent.train(n_episodes=self._n_episodes, max_episode_steps=self._n_step, render=False)

        # Assert
        self.assertEqual(self._n_episodes, len(agent.history.history))
        self.assertEqual(self._n_episodes, mocked_play_episode.call_count)
