import unittest

from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy


class TestEpsilonGreedy(unittest.TestCase):
    _sut = EpsilonGreedy
    _mock_attribute = 4

    def setUp(self) -> None:
        self._greedy_action = lambda: self._mock_attribute + 1
        self._random_action = lambda: 'random'

    def test_eps_1_returns_random_action_during_training(self):
        # Arrange
        eps = self._sut(eps_initial=1)

        # Act
        selection = eps.select(greedy_option=self._greedy_action,
                               random_option=self._random_action,
                               training=True)

        # Assert
        self.assertEqual('random', selection)

    def test_eps_1_returns_greedy_action_if_not_training(self):
        # Arrange
        eps = self._sut(eps_initial=1)

        # Act
        selection = eps.select(greedy_option=self._greedy_action,
                               random_option=self._random_action)

        # Assert
        self.assertEqual(5, selection)

    def test_eps_0_returns_greedy_action(self):
        # Arrange
        eps = self._sut(eps_initial=0)

        # Act
        selection = eps.select(greedy_option=self._greedy_action,
                               random_option=self._random_action)

        # Assert
        self.assertEqual(5, selection)

    def test_training_selection_decays_eps(self):
        # Arrange
        initial_eps = 0.99
        eps = self._sut(eps_initial=initial_eps)

        # Act
        _ = eps.select(greedy_option=self._greedy_action,
                       random_option=self._random_action,
                       training=True)

        # Assert
        self.assertLess(eps.eps_current, initial_eps)

    def test_non_training_selection_doesnt_decay_eps(self):
        # Arrange
        initial_eps = 0.99
        eps = self._sut(eps_initial=initial_eps)

        # Act
        _ = eps.select(greedy_option=self._greedy_action,
                       random_option=self._random_action,
                       training=False)

        # Assert
        self.assertAlmostEqual(eps.eps_current, initial_eps)

    def test_training_call_with_0_decay_doesnt_decay_eps(self):
        # Arrange
        initial_eps = 0.99
        eps = self._sut(eps_initial=initial_eps,
                        decay=0)

        # Act
        _ = eps.select(greedy_option=self._greedy_action,
                       random_option=self._random_action,
                       training=True)

        # Assert
        self.assertAlmostEqual(eps.eps_current, initial_eps)
