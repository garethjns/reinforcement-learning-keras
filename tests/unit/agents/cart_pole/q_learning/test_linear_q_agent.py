import copy
from typing import List

import numpy as np
from numpy.testing import assert_array_almost_equal

from agents.cart_pole.q_learning.linear_q_agent import LinearQAgent
from tests.unit.agents.cart_pole.random.test_random_agent import TestRandomAgent


class TestLinearQAgent(TestRandomAgent):
    _sut = LinearQAgent

    def _agent_specific_set_up(self):
        """This agent requires no specific set up, however this method should still be set."""
        pass

    @staticmethod
    def _checkpoint_model(agent: LinearQAgent) -> List[np.ndarray]:
        """Get coefs from each model"""
        return [copy.deepcopy(m.coef_) for m in agent.mods.values()]

    def _assert_model_unchanged(self, agent: LinearQAgent, checkpoint: List[np.ndarray]):
        for previous_coefs, current_coefs in zip([m.coef_ for m in agent.mods.values()], checkpoint):
            assert_array_almost_equal(previous_coefs, current_coefs)

    def _assert_model_changed(self, agent: LinearQAgent, checkpoint: List[np.ndarray]):
        changes_to_single_action_model = []
        for previous_coefs, current_coefs in zip([m.coef_ for m in agent.mods.values()], checkpoint):
            changes_to_single_action_model.append(np.any(np.not_equal(previous_coefs.round(8),
                                                                      current_coefs.round(8))))

        self.assertTrue(np.any(changes_to_single_action_model))

    def _assert_agent_ready(self, agent: LinearQAgent) -> None:
        self.assertIsNotNone(agent._env)

    def test_model_set_during_init(self):
        # Act
        agent = self._sut()

        # Assert
        self.assertIsInstance(agent.mods, dict)
        self.assertIsNotNone(agent.mods[0])
        self.assertIsNotNone(agent.mods[1])