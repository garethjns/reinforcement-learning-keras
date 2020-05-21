import unittest
from typing import List

from agents.cart_pole.q_learning.linear_q_agent import LinearQLearningAgent


class TestLinearQLearningAgent(unittest.TestCase):
    _sut = LinearQLearningAgent
    _created_files: List[str] = []

    def test_linear_agent_example(self):
        agent = self._sut.example(20, render=False)

        # Assert
        self.assertEqual(20, len(agent.history.history))
