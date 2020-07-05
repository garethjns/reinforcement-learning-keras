import copy
from typing import List

import numpy as np
from numpy.testing import assert_array_almost_equal

from reinforcement_learning_keras.agents.components.helpers.virtual_gpu import VirtualGPU
from reinforcement_learning_keras.agents.policy_gradient.reinforce_agent import ReinforceAgent
from reinforcement_learning_keras.enviroments.cart_pole.cart_pole_config import CartPoleConfig
from tests.unit.agents.random.test_random_agent import TestRandomAgent


class TestReinforceAgent(TestRandomAgent):
    _sut = ReinforceAgent
    _config = CartPoleConfig(agent_type='reinforce', plot_during_training=False)

    def _agent_specific_set_up(self):
        VirtualGPU(256)

        # For this agent, this call is made once at the end of the episode not on every step
        self._expected_model_update_during_training_episode: int = 0
        # This is inside the play episode function, which is still 0 here as it's called in .train
        self._expected_model_update_after_playing_episode: int = 0

    @staticmethod
    def _checkpoint_model(agent: ReinforceAgent) -> List[np.ndarray]:
        """Get coefs from each model"""
        return copy.deepcopy(agent._model.get_weights())

    def _assert_model_unchanged(self, agent: ReinforceAgent, checkpoint: List[np.ndarray]):
        model_weights = agent._model.get_weights()
        for w in range(len(model_weights)):
            assert_array_almost_equal(model_weights[w], checkpoint[w])

    def _assert_relevant_play_episode_change(self, agent: ReinforceAgent, checkpoint: List[np.ndarray]) -> None:
        self._assert_buffer_changed(agent, checkpoint)

    def _assert_relevant_after_play_episode_change(self, agent: ReinforceAgent, checkpoint: List[np.ndarray]) -> None:
        self._assert_model_changed(agent, checkpoint)

    def _assert_buffer_changed(self, agent: ReinforceAgent, checkpoint: List[np.ndarray]):
        # TODO
        pass

    def _assert_model_changed(self, agent: ReinforceAgent, checkpoint: List[np.ndarray]) -> None:
        model_weights = agent._model.get_weights()
        for w in range(len(checkpoint)):
            self.assertFalse(np.all(model_weights[w].round(6) == checkpoint[w].round(6)))

    def _assert_agent_unready(self, agent: ReinforceAgent) -> None:
        self.assertIsNone(agent._model)
        self.assertFalse(agent.ready)

    def _assert_agent_ready(self, agent: ReinforceAgent) -> None:
        self.assertIsNotNone(agent.env_builder._env)
        self.assertIsNotNone(agent._model)
        self.assertTrue(agent.ready)

    def test_train_calls_after_episode_updates_model_as_expected(self) -> None:
        # Arrange
        agent = self._ready_agent()
        checkpoint = self._checkpoint_model(agent)

        # Act
        agent.train(n_episodes=self._n_episodes, max_episode_steps=self._n_step, render=False, checkpoint_every=0)

        # Assert
        self._assert_relevant_after_play_episode_change(agent, checkpoint)


# Prevent parent test from being collected again
del TestRandomAgent
