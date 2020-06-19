"""TODO: Not finished. Can use to check if outputs look correct manually..."""

import unittest

import numpy as np

from enviroments.pong.pong_config import PONG_ENV_STACK, PONG_ENV_DIFF


SHOW = False


class TestPongStackEnvironment(unittest.TestCase):
    _sut = PONG_ENV_STACK
    _expected_shape = (84, 84, 3)

    @staticmethod
    def _plot_obs(obs: np.ndarray, show: bool = False):
        import matplotlib.pyplot as plt

        n_buff = obs.shape[2]
        fig, ax = plt.subplots(ncols=n_buff)
        for i in range(n_buff):
            ax[i].imshow(obs[:, :, i])

        if show:
            fig.show()

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
        for _ in range(20):
            obs, reward, done, _ = self._sut.step(np.random.choice([4, 5]))

            # Manually assert
            self._plot_obs(obs,
                           show=SHOW)


class TestPongDiffEnv(TestPongStackEnvironment):
    _sut = PONG_ENV_DIFF
    _expected_shape = (84, 84, 1)

    @staticmethod
    def _plot_obs(obs: np.ndarray, show: bool = True):
        import matplotlib.pyplot as plt

        plt.imshow(obs.squeeze())

        if show:
            plt.show()
