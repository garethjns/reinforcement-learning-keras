import gc
import tempfile
import unittest
from unittest.mock import MagicMock

import tensorflow as tf

from reinforcement_learning_keras.agents.components.helpers.virtual_gpu import VirtualGPU
from reinforcement_learning_keras.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from reinforcement_learning_keras.agents.q_learning.deep_q_agent import DeepQAgent
from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from reinforcement_learning_keras.enviroments.atari.pong.pong_config import PongConfig
from reinforcement_learning_keras.enviroments.cart_pole.cart_pole_config import CartPoleConfig
from reinforcement_learning_keras.enviroments.mountain_car.mountain_car_config import MountainCarConfig


class TestDeepQAgent(unittest.TestCase):
    _sut = DeepQAgent
    _fn = 'test_dqn_save.agents'
    _gpu = VirtualGPU(1024)

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()
        self._tmp_dir.cleanup()

    @staticmethod
    def _build_mock_config(base_config: PongConfig) -> MagicMock:
        config = base_config.build()
        config['eps'] = EpsilonGreedy(eps_initial=0.5, decay=0.0001, eps_min=0.01, decay_schedule='linear')
        config['replay_buffer'] = ContinuousBuffer(buffer_size=10)
        config['replay_buffer_samples'] = 2
        mock_config = MagicMock()
        mock_config.gpu_memory = 2048
        mock_config.build.return_value = config

        return mock_config

    def test_saving_and_reloading_creates_identical_object(self):
        # Arrange
        agent = self._sut(**CartPoleConfig(agent_type='dqn', plot_during_training=False,
                                           folder=self._tmp_dir.name).build())
        agent.train(verbose=True, render=False, n_episodes=2)

        # Act
        agent.save()
        agent_2 = self._sut.load(f"{agent.name}_{agent.env_spec}")
        agent_2.check_ready()

        # Assert
        self.assertEqual(agent, agent_2)

    def test_dqn_cart_pole_example(self):
        # Arrange
        config = CartPoleConfig(agent_type='dqn', plot_during_training=False,
                                folder=self._tmp_dir.name)

        # Act
        agent = self._sut.example(config, render=False, n_episodes=10)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_dqn_mountain_car_example(self):
        # Arrange
        config = MountainCarConfig(agent_type='dqn', plot_during_training=False,
                                   folder=self._tmp_dir.name)

        # Act
        agent = self._sut.example(config, render=False, max_episode_steps=50, n_episodes=10)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_dqn_pong_diff_example(self):
        # Arrange
        mock_config = self._build_mock_config(PongConfig(agent_type='dqn', mode='diff', plot_during_training=False,
                                                         folder=self._tmp_dir.name))

        # Act
        # Needs to run for long enough to fill replay buffer
        agent = self._sut.example(mock_config, render=False, max_episode_steps=20, n_episodes=3)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_dqn_pong_stack_example(self):
        # Arrange
        mock_config = self._build_mock_config(PongConfig(agent_type='dqn', mode='stack', plot_during_training=False,
                                                         folder=self._tmp_dir.name))

        # Act
        # Needs to run for long enough to fill replay buffer
        agent = self._sut.example(mock_config, render=False, max_episode_steps=20, n_episodes=3)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_dueling_dqn_cart_pole_example(self):
        # Arrange
        config = CartPoleConfig(agent_type='dueling_dqn', plot_during_training=False,
                                folder=self._tmp_dir.name)

        # Act
        agent = self._sut.example(config, render=False, n_episodes=18)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_dueling_dqn_mountain_car_example(self):
        # Arrange
        config = MountainCarConfig(agent_type='dueling_dqn', plot_during_training=False,
                                   folder=self._tmp_dir.name)

        # Act
        agent = self._sut.example(config, render=False, max_episode_steps=100, n_episodes=18)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_double_dqn_cart_pole_example(self):
        # Arrange
        config = CartPoleConfig(agent_type='double_dqn', plot_during_training=False,
                                folder=self._tmp_dir.name)

        # Act
        agent = self._sut.example(config, render=False, n_episodes=16)

        # Assert
        self.assertIsInstance(agent, self._sut)

    def test_double_dueling_dqn_cart_pole_example(self):
        # Arrange
        config = CartPoleConfig(agent_type='double_dueling_dqn', plot_during_training=False,
                                folder=self._tmp_dir.name)

        # Act
        agent = self._sut.example(config, render=False, n_episodes=20)

        # Assert
        self.assertIsInstance(agent, self._sut)
