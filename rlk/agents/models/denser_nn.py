from typing import Tuple

from tensorflow import keras

from rlk.agents.models.model_base import ModelBase


class DenserNN(ModelBase):

    def _model_architecture(self) -> Tuple[keras.layers.Layer, keras.layers.Layer]:
        frame_input = keras.layers.Input(name='input', shape=self.observation_shape)
        flat = keras.layers.Flatten(name='flatten')(frame_input)
        fc1 = keras.layers.Dense(int(flat.shape[1] / 1), name='fc1', activation='tanh')(flat)
        fc2 = keras.layers.Dense(int(flat.shape[1] / 0.5), name='fc2', activation='tanh')(fc1)
        fc3 = keras.layers.Dense(int(flat.shape[1] / 0.5), name='fc3', activation='tanh')(fc2)
        fc4 = keras.layers.Dense(int(flat.shape[1] / 1), name='fc4', activation='tanh')(fc3)
        fc5 = keras.layers.Dense(int(flat.shape[1] / 2), name='fc5', activation='tanh')(fc4)
        fc6 = keras.layers.Dense(int(flat.shape[1] / 3), name='fc6', activation='relu')(fc5)
        fc7 = keras.layers.Dense(self.n_actions * 2, name='fc7', activation='relu')(fc6)

        action_output = self._add_output(input_layer=fc7)

        return frame_input, action_output


if __name__ == "__main__":
    ma = DenserNN(observation_shape=(115,), n_actions=19, learning_rate=0.0005, output_activation='softmax')
    ma.plot()

