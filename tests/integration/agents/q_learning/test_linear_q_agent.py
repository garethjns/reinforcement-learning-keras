import tempfile
import unittest

import numpy as np

from rlk.agents.q_learning.linear_q_agent import LinearQAgent
from rlk.environments.cart_pole.cart_pole_config import CartPoleConfig
from rlk.environments.mountain_car.mountain_car_config import MountainCarConfig

try:
    from gfootball.env.config import Config
    from gfootball.env.football_env import FootballEnv
    from rlk.environments.gfootball.gfootball_config import GFootballConfig
    from rlk.environments.gfootball.register_environments import register_all

    GFOOTBALL_AVAILABLE = True
except ImportError:
    GFOOTBALL_AVAILABLE = False


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

    @unittest.skipUnless(GFOOTBALL_AVAILABLE, "GFootball not available in this env.")
    def test_gfootball_example(self):
        # Arrange
        register_all()
        config = GFootballConfig(agent_type=self._agent_type, plot_during_training=False,
                                 folder=self._tmp_dir.name)

        # Act
        agent = self._sut.example(config, render=False, max_episode_steps=100, n_episodes=20)

        # Assert
        self.assertIsInstance(agent, self._sut)
