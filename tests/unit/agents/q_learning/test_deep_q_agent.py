import copy
from typing import List
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_almost_equal

from reinforcement_learning_keras.agents.components.helpers.virtual_gpu import VirtualGPU
from reinforcement_learning_keras.agents.q_learning.deep_q_agent import DeepQAgent
from reinforcement_learning_keras.enviroments.cart_pole.cart_pole_config import CartPoleConfig
from tests.unit.agents.random.test_random_agent import TestRandomAgent


class TestDeepQAgent(TestRandomAgent):
    _sut = DeepQAgent
    _config = CartPoleConfig(agent_type='dqn', plot_during_training=False)

    def _agent_specific_set_up(self):
        VirtualGPU(256)

        # (in .play_episode)
        self._expected_model_update_after_training_episode: int = 0
        self._expected_model_update_after_playing_episode: int = 0

    def _ready_agent(self) -> DeepQAgent:
        agent = self._sut(**self._config.build())
        # Sync weights so checks are simpler
        agent.update_target_model()

        return agent

    @staticmethod
    def _checkpoint_model(agent: DeepQAgent) -> List[np.ndarray]:
        """Get coefs from each model"""
        return copy.deepcopy(agent._action_model.get_weights())

    def _assert_model_unchanged(self, agent: DeepQAgent, checkpoint: List[np.ndarray]) -> None:
        action_weights = agent._action_model.get_weights()
        value_weights = agent._target_model.get_weights()
        for w in range(len(action_weights)):
            assert_array_almost_equal(action_weights[w], checkpoint[w])
            assert_array_almost_equal(value_weights[w], checkpoint[w])

    def _assert_relevant_play_episode_change(self, agent: DeepQAgent, checkpoint: List[np.ndarray]) -> None:
        self._assert_buffer_changed(agent, checkpoint)

    def _assert_relevant_after_play_episode_change(self, agent: DeepQAgent, checkpoint: List[np.ndarray]) -> None:
        """TODO: self._assert_value_model_changed(agent, checkpoint) (?)"""
        pass

    def _assert_buffer_changed(self, agent: DeepQAgent, checkpoint: List[np.ndarray]) -> None:
        # TODO
        pass

    def _assert_action_model_changed(self, agent: DeepQAgent, checkpoint: List[np.ndarray]) -> None:
        action_weights = agent._action_model.get_weights()
        for w in range(len(checkpoint)):
            self.assertFalse(np.all(action_weights[w].round(6) == checkpoint[w].round(6)))

    def _assert_value_model_changed(self, agent: DeepQAgent, checkpoint: List[np.ndarray]) -> None:
        value_weights = agent._target_model.get_weights()
        for w in range(len(checkpoint)):
            self.assertFalse(np.all(value_weights[w].round(6) == checkpoint[w].round(6)))

    def _assert_model_changed(self, agent: DeepQAgent, checkpoint: List[np.ndarray]) -> None:
        self._assert_action_model_changed(agent, checkpoint)
        self._assert_value_model_changed(agent, checkpoint)

    def _assert_agent_unready(self, agent: DeepQAgent) -> None:
        self.assertIsNone(agent._action_model)
        self.assertIsNone(agent._target_model)
        self.assertIsNone(agent.replay_buffer)
        self.assertFalse(agent.ready)

    def _assert_agent_ready(self, agent: DeepQAgent) -> None:
        self.assertIsNotNone(agent._action_model)
        self.assertIsNotNone(agent._target_model)
        self.assertIsNotNone(agent.replay_buffer)
        self.assertTrue(agent.ready)

    def test_play_episode_steps_does_not_call_update_models_when_not_training(self) -> None:
        # Arrange
        agent = self._ready_agent()

        # Act
        with patch.object(agent, 'update_model') as mocked_update_model, \
                patch.object(agent, "update_target_model") as mocked_update_value_model:
            _ = agent.play_episode(max_episode_steps=self._n_step, training=False, render=False)

        # Assert
        self.assertEqual(self._expected_model_update_during_playing_episode, mocked_update_model.call_count)
        self.assertEqual(self._expected_model_update_after_playing_episode, mocked_update_value_model.call_count)
        self.assertEqual(self._n_step, agent.env_builder._env._max_episode_steps)

    def test_play_episode_steps_calls_update_models_when_training(self) -> None:
        # Arrange
        agent = self._ready_agent()

        # Act
        with patch.object(agent, 'update_model') as mocked_update_model, \
                patch.object(agent, "update_target_model") as mocked_update_value_model:
            _ = agent.play_episode(max_episode_steps=self._n_step, training=True, render=False)

        # Assert
        self.assertEqual(self._expected_model_update_during_training_episode, mocked_update_model.call_count)
        self.assertEqual(self._expected_model_update_after_training_episode, mocked_update_value_model.call_count)
        self.assertEqual(self._n_step, agent.env_builder._env._max_episode_steps)


del TestRandomAgent
