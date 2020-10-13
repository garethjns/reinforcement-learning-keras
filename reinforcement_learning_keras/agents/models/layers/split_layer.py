from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras


class SplitLayer(keras.layers.Layer):
    def __init__(self, split_dim: int = 3) -> None:
        """
        :param split_dim: Dimension to split array on.
        """
        super().__init__()
        self.split_dim = split_dim

    def call(self, inputs: tf.Tensor, **kwargs) -> List[tf.Tensor]:
        return [tf.expand_dims(inputs[..., i], self.split_dim) for i in range(inputs.shape[self.split_dim])]


if __name__ == "__main__":
    output = SplitLayer(split_dim=3)(np.ones(shape=(1, 3, 3, 4)))
