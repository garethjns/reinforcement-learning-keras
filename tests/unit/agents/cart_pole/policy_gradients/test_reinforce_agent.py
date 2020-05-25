import copy
from typing import List

import numpy as np
from numpy.testing import assert_array_almost_equal

from agents.cart_pole.policy_gradient.reinforce_agent import ReinforceAgent
from tests.unit.agents.cart_pole.random.test_random_agent import TestRandomAgent


class TestReinforceAgent(TestRandomAgent):
    _sut = ReinforceAgent

    def _agent_specific_set_up(self):
        self._sut.set_tf(256)

        # For this agent, this call is made once at the end of the episode not on every step
        self._expected_model_update_during_training_episode: int = 1
        self._expected_model_update_after_playing_episode: int = 0

    @staticmethod
    def _checkpoint_model(agent: ReinforceAgent) -> List[np.ndarray]:
        """Get coefs from each model"""
        return copy.deepcopy(agent._model.get_weights())

    def _assert_model_unchanged(self, agent: ReinforceAgent, checkpoint: List[np.ndarray]):
        model_weights = agent._model.get_weights()
        for w in range(len(model_weights)):
            assert_array_almost_equal(model_weights[w], checkpoint[w])

    def _assert_model_changed(self, agent: ReinforceAgent, checkpoint: List[np.ndarray]):
        model_weights = agent._model.get_weights()
        for w in range(len(checkpoint)):
            self.assertFalse(np.all(model_weights[w].round(6) == checkpoint[w].round(6)))

    def _assert_agent_unready(self, agent: ReinforceAgent) -> None:
        self.assertIsNone(agent._env)
        self.assertIsNone(agent._model)
        self.assertIsNotNone(agent._model_weights)

    def _assert_agent_ready(self, agent: ReinforceAgent) -> None:
        self.assertIsNotNone(agent._env)
        self.assertIsNotNone(agent._model)
        self.assertIsNone(agent._model_weights)
