from typing import Tuple

from sklearn.base import BaseEstimator, TransformerMixin


class Clipper(BaseEstimator, TransformerMixin):
    lim: Tuple[float, float]

    def __init__(self, lim: Tuple[float, float] = (-1, 1)):
        self.set_params(lim=lim)

    def fit(self, x=None, y=None):
        return self

    def transform(self, x):
        x[x < self.lim[0]] = self.lim[0]
        x[x > self.lim[1]] = self.lim[1]
        return x
