from typing import Tuple

from tensorflow import keras
from tensorflow.keras import backend as K

from reinforcement_learning_keras.enviroments.model_base import ModelBase


class SmallDuelingNN(ModelBase):

    def _model_architecture(self) -> Tuple[keras.layers.Layer, keras.layers.Layer]:
        n_units = 16 * self.unit_scale

        state_input = keras.layers.Input(name='input', shape=self.observation_shape)
        fc1 = keras.layers.Dense(units=int(n_units), name='fc1', activation='relu')(state_input)
        fc2 = keras.layers.Dense(units=int(n_units / 2), name='fc2', activation='relu')(fc1)

        # Separate layers for baseline value (1 node) and action advantages (n action nodes)
        v_layer = keras.layers.Dense(1, activation='linear')(fc2)
        a_layer = keras.layers.Dense(self.n_actions, activation='linear')(fc2)

        # Function definition for lambda layer. Base value layer + actions value layer - mean action values
        def _merge_layer(layer_inputs: list):
            return layer_inputs[0] + layer_inputs[1] - K.mean(layer_inputs[1], axis=1, keepdims=True)

        action_output = keras.layers.Lambda(_merge_layer, output_shape=(self.n_actions,),
                                            name="output")([v_layer, a_layer])

        return state_input, action_output
