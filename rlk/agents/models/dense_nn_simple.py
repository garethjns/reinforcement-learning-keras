from typing import Tuple, List

from tensorflow import keras

from rlk.agents.models.model_base import ModelBase


class DenseNNSimple(ModelBase):

    def _model_architecture(self) -> Tuple[List[keras.layers.Layer], List[keras.layers.Layer]]:
        frame_input = keras.layers.Input(name='input', shape=self.observation_shape)
        # flat = keras.layers.Flatten(name='flatten')(frame_input)

        fc1 = keras.layers.Dense(128, name='fc1', activation=self.hidden_layer_activations)(frame_input)
        action_output = self._add_output(input_layer=fc1)

        return [frame_input], action_output
