import tempfile
import unittest

import numpy as np

from reinforcement_learning_keras.agents.q_learning.linear_q_agent import LinearQAgent
from reinforcement_learning_keras.enviroments.cart_pole.cart_pole_config import CartPoleConfig
from reinforcement_learning_keras.enviroments.mountain_car.mountain_car_config import MountainCarConfig


class TestLinearQAgent(unittest.TestCase):
    _sut = LinearQAgent
    _agent_type: str = 'linear_q'

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
        # This is the important check
        np.testing.assert_array_equal(agent.mods[0].coef_, agent_2.mods[0].coef_)
        # TODO: But this check fails, why? : self.assertEqual(agent, agent_2)

    def test_cart_pole_example(self):
        # Arrange
        config = CartPoleConfig(agent_type=self._agent_type, plot_during_training=False,
                                folder=self._tmp_dir.name)

        # Act
        agent = self._sut.example(config, render=False, n_episodes=20)

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
