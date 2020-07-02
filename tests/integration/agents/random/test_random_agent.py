import tempfile
import unittest

from reinforcement_learning_keras.agents.random.random_agent import RandomAgent
from reinforcement_learning_keras.enviroments.atari.pong.pong_config import PongConfig
from reinforcement_learning_keras.enviroments.cart_pole.cart_pole_config import CartPoleConfig
from reinforcement_learning_keras.enviroments.mountain_car.mountain_car_config import MountainCarConfig


class TestRandomAgent(unittest.TestCase):
    _sut = RandomAgent
    _agent_type: str = 'random'

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_saving_and_reloading_creates_identical_object(self):
        # Arrange
        agent = self._sut(**CartPoleConfig(agent_type=self._agent_type, plot_during_training=False,
                                           folder=self._tmp_dir.name).build())
        agent.train(verbose=True, render=False, n_episodes=2)

        # Act
        agent.save()
        agent_2 = self._sut.load(f"{agent.name}_{agent.env_spec}")
        agent_2.check_ready()

        # Assert
        self.assertEqual(agent, agent_2)

    def test_cart_pole_example(self):
        # Arrange
        config = CartPoleConfig(agent_type=self._agent_type, plot_during_training=False,
                                folder=self._tmp_dir.name)

        # Act
        agent = self._sut.example(config, render=False, n_episodes=200)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_mountain_car_example(self):
        # Arrange
        config = MountainCarConfig(agent_type=self._agent_type, plot_during_training=False,
                                   folder=self._tmp_dir.name)

        # Act
        agent = self._sut.example(config, render=False, max_episode_steps=100, n_episodes=20)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_pong_example(self):
        # Arrange
        config = PongConfig(agent_type=self._agent_type, plot_during_training=False,
                            folder=self._tmp_dir.name)

        # Act
        agent = self._sut.example(config, render=False, max_episode_steps=100, n_episodes=20)

        # Assert
        self.assertIsInstance(agent, self._sut)
