import unittest

import numpy as np

from agents.cart_pole.q_learning.components.replay_buffer import ReplayBuffer


class TesReplayBuffer(unittest.TestCase):
    def test_unbuffered_state_1_dim(self):
        rb = ReplayBuffer(buffer_size=50)

        _ = [rb.append((np.zeros(shape=(5,)) + i, i, i, False)) for i in range(10)]

        ss, aa, rr, dd, ss_ = rb.sample(2)

        ss_0 = [np.unique(ss[0][0])]
        ss__0 = [np.unique(ss_[0][0])]

        self.assertEqual(aa[0], int(ss_0[0]))
        self.assertEqual(rr[0], int(ss_0[0]))
        self.assertListEqual([s + 1 for s in ss_0], ss__0)
