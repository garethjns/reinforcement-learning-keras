import collections
from dataclasses import dataclass
from typing import Tuple, Any

import numpy as np


@dataclass
class ReplayBuffer:
    buffer_size: int = 50

    def __post_init__(self):
        self._state_queue = collections.deque(maxlen=self.buffer_size)
        self._other_queue = collections.deque(maxlen=self.buffer_size)

        self.queue = collections.deque(maxlen=self.buffer_size)

    def __len__(self):
        return self.n if self.n > 0 else 0

    @property
    def full(self):
        return len(self._state_queue) == self.buffer_size

    @property
    def n(self) -> int:
        return len(self._state_queue) - 1

    def append(self, items: Tuple[Any, int, float, bool]):
        """
        :param items: Tuple containing (s, a, r, d).
        """
        self._state_queue.append(items[0])
        self._other_queue.append(items[1::])

    def sample(self, n: int):
        if n > self.n:
            raise ValueError

        idx = np.random.randint(0, self.n, n)

        ss = [self._state_queue[i] for i in idx]
        ss_ = [self._state_queue[i + 1] for i in idx]

        ard = [self._other_queue[i] for i in idx]
        aa = [a for (a, _, _) in ard]
        rr = [r for (_, r, _) in ard]
        dd = [d for (_, _, d) in ard]

        return ss, aa, rr, dd, ss_
