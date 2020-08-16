import collections
from dataclasses import dataclass
from typing import Tuple, Any, Iterable, List

import numpy as np

from reinforcement_learning_keras.agents.components.replay_buffers.replay_buffer_base import ReplayBufferBase


@dataclass
class ContinuousBuffer(ReplayBufferBase):
    buffer_size: int = 50

    def __post_init__(self) -> None:
        self._state_queue = collections.deque(maxlen=self.buffer_size)
        self._other_queue = collections.deque(maxlen=self.buffer_size)

        self.queue = collections.deque(maxlen=self.buffer_size)

    def __len__(self) -> int:
        return self.n if (self.n > 0) else 0

    @property
    def full(self) -> bool:
        return len(self._state_queue) == self.buffer_size

    @property
    def n(self) -> int:
        return len(self._state_queue) - 1

    def append(self, items: Tuple[Any, int, float, bool]) -> None:
        """
        :param items: Tuple containing (s, a, r, d).
        """
        self._state_queue.append(items[0])
        self._other_queue.append(items[1::])

    def get_batch(self, idxs: Iterable[int]) -> Tuple[List[np.ndarray], List[np.ndarray],
                                                      List[float], List[bool], List[np.ndarray]]:
        ss = [self._state_queue[i] for i in idxs]
        ss_ = [self._state_queue[i + 1] for i in idxs]

        ard = [self._other_queue[i] for i in idxs]
        aa = [a for (a, _, _) in ard]
        rr = [r for (_, r, _) in ard]
        dd = [d for (_, _, d) in ard]

        return ss, aa, rr, dd, ss_

    def sample_batch(self, n: int) -> Tuple[List[np.ndarray], List[np.ndarray],
                                            List[float], List[bool], List[np.ndarray]]:
        if n > self.n:
            raise ValueError

        idxs = np.random.randint(0, self.n, n)
        return self.get_batch(idxs)
