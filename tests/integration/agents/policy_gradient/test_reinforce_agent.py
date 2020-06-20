import unittest

from agents.components.helpers.virtual_gpu import VirtualGPU
from agents.policy_gradient.reinforce_agent import ReinforceAgent
from enviroments.cart_pole.cart_pole_config import CartPoleConfig


class TestReinforceAgent(unittest.TestCase):
    _sut = ReinforceAgent
    _agent_type: str = 'reinforce'
    _gpu = VirtualGPU(256)

    def test_saving_and_reloading_creates_identical_object(self):
        # Arrange
        agent = self._sut(**CartPoleConfig(agent_type=self._agent_type, plot_during_training=False).build())
        agent.train(verbose=True, render=False, n_episodes=2)

        # Act
        agent.save()
        agent_2 = self._sut.load(f"{agent.name}_{agent.env_spec}")
        agent_2.check_ready()

        # Assert
        self.assertEqual(agent, agent_2)

    def test_cart_pole_example(self):
        # Arrange
        config = CartPoleConfig(agent_type=self._agent_type, plot_during_training=False)

        # Act
        agent = self._sut.example(config, render=False, n_episodes=10)

        # Assert
        self.assertIsInstance(agent, self._sut)
