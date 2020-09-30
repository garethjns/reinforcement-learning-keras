from typing import Tuple

from tensorflow import keras

from reinforcement_learning_keras.enviroments.model_base import ModelBase


class DenseNN(ModelBase):
    """A convolutional NN for Pong, similar to Google paper."""

    def _model_architecture(self) -> Tuple[keras.layers.Layer, keras.layers.Layer]:
        frame_input = keras.layers.Input(name='input', shape=self.observation_shape)
        fc1 = keras.layers.Dense(int(self.observation_shape[0] / 1.5), name='fc1', activation='relu')(frame_input)
        fc2 = keras.layers.Dense(int(self.observation_shape[0] / 3), name='fc2', activation='relu')(fc1)
        fc3 = keras.layers.Dense(self.n_actions * 2, name='fc3', activation='relu')(fc2)
        action_output = keras.layers.Dense(units=self.n_actions, name='output',
                                           activation=self.output_activation)(fc3)

        return frame_input, action_output
