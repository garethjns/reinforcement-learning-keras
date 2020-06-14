import abc
from dataclasses import dataclass
from typing import Tuple, Union, Callable, List

from tensorflow import keras


@dataclass
class ModelBase(abc.ABC):
    def __init__(self, observation_shape: List[int], n_actions: int, output_activation: Union[None, str] = None,
                 unit_scale: int = 1, opt: Union[keras.optimizers.Optimizer, None] = None,
                 learning_rate: float = 0.0001, loss: Union[str, Callable] = 'mse'):
        """
        :param observation_shape: Tuple specifying input shape.
        :param n_actions: Int specifying number of outputs
        :param output_activation: Activation function for output. Eg. None for value estimation (off-policy methods) or
                                  'softmax' for action probabilities (on-policy methods).
        :param unit_scale: Multiplier for all units in FC layers in network. Default 1 = 16 units for first layer,
                            8 for second.
        :param opt: Keras optimiser to use. Default  keras.optimizers.Adam(learning_rate=0.0001).
        :param learning_rate: Learning rate for optimiser, only in construction of default ADAM opt if no opt specified.
        :param loss: Model loss. Default 'mse'. Can be custom callable.
        """
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.unit_scale = unit_scale
        self.output_activation = output_activation
        self.loss = loss
        if opt is None:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        self.learning_rate = opt.learning_rate.numpy()
        self.opt = opt

    def compile(self, model_name: str) -> keras.Model:
        """Compile a copy of the model."""

        state_input, action_output = self._model_architecture()
        model = keras.Model(inputs=[state_input], outputs=[action_output], name=model_name)
        model.compile(self.opt, loss=self.loss)

        return model

    @abc.abstractmethod
    def _model_architecture(self) -> Tuple[keras.layers.Layer, keras.layers.Layer]:
        """Define model construction function. Should return input layer and output layer."""
        pass

    def plot(self):
        """TODO."""
        pass
