import gc
import tempfile
import unittest
from unittest.mock import MagicMock

import tensorflow as tf
from tf2_vgpu import VirtualGPU

from rlk.agents.actor_critic.actor_critic import ActorCriticAgent
from rlk.environments.atari.pong.pong_config import PongConfig
from rlk.environments.cart_pole.cart_pole_config import CartPoleConfig
from rlk.environments.mountain_car.mountain_car_config import MountainCarConfig


class TestActorCriticAgent(unittest.TestCase):
    _sut = ActorCriticAgent
    _fn = 'test_ac_save.agents'
    _gpu = VirtualGPU(1024)

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()

        # Ensure eager model is on, this must be done in each setup.
        # (TODO: But why? It's on by default and is only turned off in DeepQAgent and ReinforceAgent.
        #  Perhaps an import in Configs?)
        tf.compat.v1.enable_eager_execution()

    def tearDown(self):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()
        self._tmp_dir.cleanup()

    @staticmethod
    def _build_mock_config(base_config: PongConfig) -> MagicMock:
        config = base_config.build()
        mock_config = MagicMock()
        mock_config.gpu_memory = 2048
        mock_config.build.return_value = config

        return mock_config

    def test_saving_and_reloading_creates_identical_object(self):
        # Arrange
        agent = self._sut(**CartPoleConfig(agent_type='actor_critic', plot_during_training=False,
                                           folder=self._tmp_dir.name).build())
        agent.train(verbose=True, render=False, n_episodes=2)

        # Act
        agent.save()
        agent_2 = self._sut.load(f"{agent.name}_{agent.env_spec}")
        agent_2.check_ready()

        # Assert
        self.assertEqual(agent, agent_2)

    def test_ac_cart_pole_example(self):
        # Arrange
        config = CartPoleConfig(agent_type='actor_critic', plot_during_training=False,
                                folder=self._tmp_dir.name)

        # Act
        agent = self._sut.example(config, render=False, n_episodes=10)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_ac_mountain_car_example(self):
        # Arrange
        config = MountainCarConfig(agent_type='actor_critic', plot_during_training=False,
                                   folder=self._tmp_dir.name)

        # Act
        agent = self._sut.example(config, render=False, max_episode_steps=50, n_episodes=10)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_ac_pong_diff_example(self):
        # Arrange
        mock_config = self._build_mock_config(
            PongConfig(agent_type='actor_critic', mode='diff', plot_during_training=False,
                       folder=self._tmp_dir.name))

        # Act
        # Needs to run for long enough to fill replay buffer
        agent = self._sut.example(mock_config, render=False, max_episode_steps=20, n_episodes=3)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_ac_pong_stack_example(self):
        # Arrange
        mock_config = self._build_mock_config(
            PongConfig(agent_type='actor_critic', mode='stack', plot_during_training=False,
                       folder=self._tmp_dir.name))

        # Act
        # Needs to run for long enough to fill replay buffer
        agent = self._sut.example(mock_config, render=False, max_episode_steps=20, n_episodes=3)

        # Assert
        self.assertIsInstance(agent, self._sut)
