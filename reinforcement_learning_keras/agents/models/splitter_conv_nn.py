from typing import Tuple, List, Iterable

from tensorflow import keras

from reinforcement_learning_keras.agents.models.layers.split_layer import SplitLayer
from reinforcement_learning_keras.agents.models.model_base import ModelBase


class SplitterConvNN(ModelBase):
    def __init__(self, *args, additional_dense_input_shape: Tuple[int, ...] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.additional_dense_input_shape = additional_dense_input_shape

    @staticmethod
    def _build_conv_branch(frame: keras.layers.Layer, name: str) -> keras.layers.Layer:
        conv1 = keras.layers.Conv2D(16, kernel_size=(8, 8), strides=(4, 4),
                                    name=f'conv1_frame_{name}', padding='same',
                                    activation='relu')(frame)
        mp1 = keras.layers.MaxPooling2D(pool_size=2, name=f"mp1_frame_{name}")(conv1)
        conv2 = keras.layers.Conv2D(24, kernel_size=(4, 4), strides=(2, 2),
                                    name=f'conv2_frame_{name}', padding='same',
                                    activation='relu')(mp1)
        mp2 = keras.layers.MaxPooling2D(pool_size=2, name=f"mp2_frame_{name}")(conv2)
        conv3 = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                    name=f'conv3_frame_{name}', padding='same',
                                    activation='relu')(mp2)
        mp3 = keras.layers.MaxPooling2D(pool_size=2, name=f"mp3_frame_{name}")(conv3)

        flatten = keras.layers.Flatten(name=f'flatten_{name}')(mp3)

        return flatten

    def _model_architecture(self) -> Tuple[List[keras.layers.Layer], keras.layers.Layer]:
        n_units = 512 * self.unit_scale

        frames_input = keras.layers.Input(name='conv_input', shape=self.observation_shape)
        frames_split = SplitLayer(split_dim=3)(frames_input)
        conv_branches = []
        for f, frame in enumerate(frames_split):
            conv_branches.append(self._build_conv_branch(frame, name=str(f)))

        if self.additional_dense_input_shape is not None:
            additional_dense_input = [keras.layers.Input(name='dense_input', shape=self.additional_dense_input_shape)]
        else:
            additional_dense_input = []

        concat_conv = keras.layers.Concatenate()(conv_branches + additional_dense_input)
        fc1 = keras.layers.Dense(units=int(n_units), name='fc1', activation='relu')(concat_conv)
        fc2 = keras.layers.Dense(units=int(n_units / 1.5), name='fc2', activation='relu')(fc1)
        fc3 = keras.layers.Dense(units=int(n_units / 2.5), name='fc3', activation='relu')(fc2)

        action_output = self._add_output(input_layer=fc3)

        return [frames_input] + additional_dense_input, action_output


if __name__ == "__main__":
    SplitterConvNN(observation_shape=(100, 100, 4), n_actions=3).plot('splitter')
    SplitterConvNN(observation_shape=(100, 100, 4), n_actions=3, dueling=True).plot("splitter_dueling")
    SplitterConvNN(observation_shape=(100, 100, 4), additional_dense_input_shape=(5,),
                   n_actions=3, dueling=True).plot("splitter_dense_dueling")
