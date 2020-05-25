import os
import unittest
from typing import List

from agents.cart_pole.random.random_agent import RandomAgent


class TestRandomAgent(unittest.TestCase):
    """
    The random agent is basically a non-abstract version of AgentBase. It's used here to define the general integration
    test interface, as if it's a mocked AgentBase.

    These tests are standard across agents, which just need to set the _sut accordingly, and handle calling
    Agent.set_tf in a set up set, where required.
    """

    _sut = RandomAgent
    _created_files: List[str] = []
    _fn: str = 'test_random_save.agent'

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
        self._created_files += self._fn

        # Act
        agent.save(self._fn)
        agent_2 = self._sut.load(self._fn)
        agent_2.check_ready()

        # Assert
        self.assertEqual(agent, agent_2)

    def test_reinforce_example(self):
        # Act
        agent = self._sut.example(16, render=False)

        # Assert
        self.assertEqual(16, len(agent.history.history))
