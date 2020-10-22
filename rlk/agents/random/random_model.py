import numpy as np


class RandomModel:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def predict(self) -> int:
        return int(np.random.randint(0, self.n_actions, 1))
