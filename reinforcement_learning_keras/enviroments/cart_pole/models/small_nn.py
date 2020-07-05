from typing import Tuple

from tensorflow import keras

from reinforcement_learning_keras.enviroments.model_base import ModelBase


class SmallNN(ModelBase):

    def _model_architecture(self) -> Tuple[keras.layers.Layer, keras.layers.Layer]:
        n_units = 16 * self.unit_scale

        state_input = keras.layers.Input(name='input', shape=self.observation_shape)
        fc1 = keras.layers.Dense(units=int(n_units), name='fc1', activation='relu')(state_input)
        fc2 = keras.layers.Dense(units=int(n_units / 2), name='fc2', activation='relu')(fc1)
        action_output = keras.layers.Dense(units=self.n_actions, name='output', activation=self.output_activation)(fc2)

        return state_input, action_output
