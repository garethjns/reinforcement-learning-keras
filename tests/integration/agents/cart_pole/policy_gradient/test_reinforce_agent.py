import unittest

from agents.cart_pole.policy_gradient.reinforce_agent import ReinforceAgent
from agents.agent_helpers.virtual_gpu import VirtualGPU


class TestReinforceAgent(unittest.TestCase):
    _sut = ReinforceAgent
    _fn = 'test_reinforce_save.agents'

    @classmethod
    def setUp(cls):
        VirtualGPU(256)

    def test_saving_and_reloading_creates_identical_object(self):
        # Arrange
        agent = self._sut("CartPole-v0")
        agent.train(verbose=True, render=False, n_episodes=2)

        # Act
        agent.save(self._fn)
        agent_2 = self._sut.load(self._fn)
        agent_2.check_ready()

        # Assert
        self.assertEqual(agent, agent_2)

    def test_example(self):
        # Act
        agent = self._sut.example(16, render=False)

        # Assert
        self.assertEqual(16, len(agent.history.history))
