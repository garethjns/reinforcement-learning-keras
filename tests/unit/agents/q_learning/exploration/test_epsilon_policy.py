import unittest

from rlk.agents.q_learning.exploration.epsilon_policy import EpsilonPolicy


class TestEpsilonPolicy(unittest.TestCase):
    _greedy_option = lambda s: 0
    _mock_policy = lambda s: 1
    _n = 000

    def setUp(self):
        self._sut = EpsilonPolicy(actions_pool=[0, 1], policy=self._mock_policy)

    def test_always_returns_greedy_option_with_epsilon_0(self):
        # Arrange
        self._sut.eps_current = 0

        # Act
        actions = []
        for _ in range(self._n):
            actions.append(self._sut.select(state=None, greedy_option=self._greedy_option, training=False))

        # Assert
        self.assertListEqual([self._greedy_option()] * self._n, actions)

    def test_always_returns_policy_option_with_epsilon_1(self):
        # Arrange
        self._sut.eps_current = 1

        # Act
        actions = []
        for _ in range(self._n):
            actions.append(self._sut.select(state=None, greedy_option=self._greedy_option, training=False))

        # Assert
        self.assertListEqual([self._mock_policy()] * self._n, actions)
