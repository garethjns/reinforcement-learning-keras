import unittest

from agents.components.helpers.virtual_gpu import VirtualGPU
from agents.q_learning.deep_q_agent import DeepQAgent
from enviroments.cart_pole.cart_pole_config import CartPoleConfig
from enviroments.mountain_car.mountain_car_config import MountainCarConfig
from enviroments.pong.pong_config import PongConfig


class TestDeepQAgent(unittest.TestCase):
    _sut = DeepQAgent
    _agent_type: str = 'dqn'
    _fn = 'test_dqn_save.agents'

    def setUp(self):
        VirtualGPU(256)

    def test_saving_and_reloading_creates_identical_object(self):
        # Arrange
        agent = self._sut(**CartPoleConfig(agent_type=self._agent_type, plot_during_training=False).build())
        agent.train(verbose=True, render=False, n_episodes=2)

        # Act
        agent.save(self._fn)
        agent_2 = self._sut.load(self._fn)
        agent_2.check_ready()

        # Assert
        self.assertEqual(agent, agent_2)

    def test_dqn_cart_pole_example(self):
        # Arrange
        config = CartPoleConfig(agent_type='dqn', plot_during_training=False)

        # Act
        agent = self._sut.example(config, render=False, n_episodes=10)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_dqn_mountain_car_example(self):
        # Arrange
        config = MountainCarConfig(agent_type='dqn', plot_during_training=False)

        # Act
        agent = self._sut.example(config, render=False, max_episode_steps=50, n_episodes=10)

        # Assert
        self.assertIsInstance(agent, self._sut)

    @unittest.skip("OOM error and slow...")
    def test_dqn_pong_example(self):
        # Arrange
        config = PongConfig(agent_type='dqn', plot_during_training=False)

        # Act
        # Needs to run for long enough to fill replay buffer
        agent = self._sut.example(config, render=False, max_episode_steps=1000, n_episodes=12)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_dueling_dqn_cart_pole_example(self):
        # Arrange
        config = CartPoleConfig(agent_type='dueling_dqn', plot_during_training=False)

        # Act
        agent = self._sut.example(config, render=False, n_episodes=10)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_dueling_dqn_mountain_car_example(self):
        # Arrange
        config = MountainCarConfig(agent_type='dueling_dqn', plot_during_training=False)

        # Act
        agent = self._sut.example(config, render=False, max_episode_steps=100, n_episodes=10)

        # Assert
        self.assertIsInstance(agent, self._sut)

