import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from reinforcement_learning_keras.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer


class TestContinuousBuffer(unittest.TestCase):
    _sut = ContinuousBuffer

    def _fill_replay_buffer(self, shape=(5,)) -> ContinuousBuffer:
        replay_buffer = self._sut()
        for i in range(10):
            replay_buffer.append((np.zeros(shape=shape) + i,
                                  i, 0.9, False))

        return replay_buffer

    def _fill_replay_buffer_include_dones(self, shape=(5,)) -> ContinuousBuffer:
        replay_buffer = self._sut()
        dones = np.array([False] * 10)
        dones[6] = True
        for i, d in zip(range(10), dones):
            replay_buffer.append((np.zeros(shape=shape) + i,
                                  i, 0.9, d))

        return replay_buffer

    def test_get_sample_with_1_dim_state_returns_expected_shapes(self) -> None:
        # Arrange
        shape = (5,)
        rb = self._fill_replay_buffer()

        # Act
        ss, aa, rr, dd, ss_ = rb.sample_batch(2)

        # Assert
        ss_0 = [np.unique(ss[0][0])]
        ss__0 = [np.unique(ss_[0][0])]
        self.assertListEqual([s + 1 for s in ss_0], ss__0)

        self.assertIsInstance(aa, list)
        self.assertIsInstance(rr, list)
        self.assertEqual(shape, ss[0].shape)

    def test_sequential_get_batch_returns_sequential_samples(self) -> None:
        # Arrange
        rb = self._fill_replay_buffer(shape=(5, 5, 3))

        # Act
        ss, aa, rr, dd, ss_ = rb.get_batch(idxs=[4, 5])

        # Assert
        assert_array_almost_equal(ss[1], ss_[0])

    def test_get_sample_with_multi_dim_state_returns_expected_shapes(self) -> None:
        # Arrange
        shape = (5, 5, 3)
        rb = self._fill_replay_buffer(shape=shape)

        # Act
        ss, aa, rr, dd, ss_ = rb.sample_batch(8)

        # Assert
        ss_0 = [np.unique(ss[0][0])]
        ss__0 = [np.unique(ss_[0][0])]
        self.assertListEqual([s + 1 for s in ss_0], ss__0)
        self.assertEqual(shape, ss[0].shape)
