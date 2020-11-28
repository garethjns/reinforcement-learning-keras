from typing import Tuple

from tensorflow import keras

from rlk.agents.models.model_base import ModelBase


class ConvNN(ModelBase):
    """A convolutional NN for Pong, similar to Google paper."""

    def _model_architecture(self) -> Tuple[keras.layers.Layer, keras.layers.Layer]:
        n_units = 512 * self.unit_scale

        frame_input = keras.layers.Input(name='input', shape=self.observation_shape)
        conv1 = keras.layers.Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                                    name='conv1', padding='same', activation='relu')(frame_input)
        conv2 = keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2),
                                    name='conv2', padding='same', activation='relu')(conv1)
        conv3 = keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                    name='conv3', padding='same', activation='relu')(conv2)
        flatten = keras.layers.Flatten(name='flatten')(conv3)
        fc1 = keras.layers.Dense(units=int(n_units), name='fc1', activation='relu')(flatten)

        action_output = self._add_output(input_layer=fc1)

        return frame_input, action_output


if __name__ == "__main__":
    ConvNN(observation_shape=(84, 84, 3), n_actions=6, dueling=False).plot(model_name='pong')
    ConvNN(observation_shape=(84, 84, 3), n_actions=6, dueling=True).plot(model_name='pong_dueling')
