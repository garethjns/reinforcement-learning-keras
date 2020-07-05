import time
import unittest

import matplotlib.pyplot as plt
import numpy as np

from reinforcement_learning_keras.enviroments.atari.pong.pong_config import PongConfig


class TestPongStackEnvironment(unittest.TestCase):
    _sut = PongConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (84, 84, 3)
    _n_steps = 20
    _show = False

    def _plot_obs(self, obs: np.ndarray):
        n_buff = obs.shape[2]
        fig, ax = plt.subplots(ncols=n_buff)
        for i in range(n_buff):
            ax[i].imshow(obs[:, :, i])

        if self._show:
            fig.show()
            time.sleep(0.1)

    def test_reset_returns_expected_obs_shape(self):
        # Act
        obs = self._sut.reset()

        # Assert
        self.assertEqual(self._expected_shape, obs.shape)

    def test_reset_returns_expected_obs_value(self):
        # Act
        obs = self._sut.reset()

        # Assert
        self.assertLess(obs[0, 0, 0], 1)

    def test_step_returns_expected_obs_shape(self):
        # Arrange
        _ = self._sut.reset()

        # Act
        obs, reward, done, _ = self._sut.step(0)

        # Assert
        self.assertEqual(self._expected_shape, obs.shape)

    def test_step_returns_expected_obs_value(self):
        # Arrange
        _ = self._sut.reset()

        # Act
        obs, reward, done, _ = self._sut.step(0)

        # Assert
        self.assertLess(obs[0, 0, 0], 1)

    def test_multiple_steps(self):
        # Arrange
        _ = self._sut.reset()

        # Act
        for _ in range(self._n_steps):
            obs, reward, done, _ = self._sut.step(np.random.choice(range(self._sut.action_space.n)))

            # Manually assert
            self._plot_obs(obs)


class TestPongDiffEnvironment(TestPongStackEnvironment):
    _sut = PongConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (84, 84, 1)
    _show = False

    def _plot_obs(self, obs: np.ndarray):
        plt.imshow(obs.squeeze())

        if self._show:
            plt.show()
            time.sleep(0.1)
