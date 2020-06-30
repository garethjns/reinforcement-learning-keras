from typing import Tuple

from tensorflow import keras
from tensorflow.keras import backend as K

from reinforcement_learning_keras.enviroments.model_base import ModelBase


class DuelingConvNN(ModelBase):
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
        # Separate layers for baseline value (1 node) and action advantages (n action nodes)
        v_layer = keras.layers.Dense(1, activation='linear')(fc1)
        a_layer = keras.layers.Dense(self.n_actions, activation='linear')(fc1)

        # Function definition for lambda layer. Base value layer + actions value layer - mean action values
        def _merge_layer(layer_inputs: list):
            return layer_inputs[0] + layer_inputs[1] - K.mean(layer_inputs[1], axis=1, keepdims=True)

        action_output = keras.layers.Lambda(_merge_layer, output_shape=(self.n_actions,),
                                            name="output")([v_layer, a_layer])

        return frame_input, action_output
