import os
import unittest
from typing import List

from agents.cart_pole.q_learning.linear_q_agent import LinearQAgent


class TestLinearQLearningAgent(unittest.TestCase):
    _sut = LinearQAgent
    _created_files: List[str] = []

    @classmethod
    def tearDown(cls):
        for f in cls._created_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except (PermissionError, IsADirectoryError):
                    pass

    def test_saving_and_reloading_creates_identical_object(self):
        # Arrange
        agent = self._sut("CartPole-v0")
        agent.train(verbose=True, render=False, n_episodes=2)
        fn = 'test_linear_save.agent'
        self._created_files += fn

        # Act
        agent.save(fn)
        agent_2 = self._sut.load(fn)
        agent_2.check_ready()

        # Assert
        self.assertEqual(agent, agent_2)

    def test_linear_agent_example(self):
        agent = self._sut.example(20, render=False)

        # Assert
        self.assertEqual(20, len(agent.history.history))
