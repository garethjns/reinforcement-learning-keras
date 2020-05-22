import os
import unittest
from typing import List

from agents.cart_pole.policy_gradient.reinforce_agent import ReinforceAgent


class TestReinforceAgent(unittest.TestCase):
    _sut = ReinforceAgent
    _created_files: List[str] = []

    @classmethod
    def setUp(cls):
        cls._sut.set_tf(256)

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
        fn = 'test_reinforce_save.agent'
        self._created_files += fn

        # Act
        agent.save(fn)
        agent_2 = self._sut.load(fn)
        agent_2.check_ready()

        # Assert
        self.assertEqual(agent, agent_2)

    def test_reinforce_example(self):
        # Act
        agent = self._sut.example(16, render=False)

        # Assert
        self.assertEqual(16, len(agent.history.history))
