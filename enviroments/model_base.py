import abc
from dataclasses import dataclass
from typing import Tuple, Union, Callable, List

from tensorflow import keras


@dataclass
class ModelBase(abc.ABC):
    def __init__(self, observation_shape: List[int], n_actions: int, output_activation: Union[None, str] = None,
                 unit_scale: int = 1, learning_rate: float = 0.0001, opt: str = 'Adam') -> None:
        """
        :param observation_shape: Tuple specifying input shape.
        :param n_actions: Int specifying number of outputs
        :param output_activation: Activation function for output. Eg. None for value estimation (off-policy methods) or
                                  'softmax' for action probabilities (on-policy methods).
        :param unit_scale: Multiplier for all units in FC layers in network. Default 1 = 16 units for first layer,
                            8 for second.
        :param opt: Keras optimiser to use. Should be string. This is to avoid storing TF/Keras objects here.
        :param learning_rate: Learning rate for optimiser.

        """
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.unit_scale = unit_scale
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.opt = opt

    def compile(self, model_name: str = 'model', loss: Union[str, Callable] = 'mse') -> keras.Model:
        """
        Compile a copy of the model using the provided loss.

        Note loss is added here to avoid storing in self. We don't want to do that, as if this model is pickled deepcopy
        will be disabled for TF objects if eager mode is disabled. It's better to use as needed rather than storing.

        :param model_name: Name of model
        :param loss: Model loss. Default 'mse'. Can be custom callable.
        """
        # Get optimiser
        if self.opt.lower() == 'adam':
            opt = keras.optimizers.Adam
        elif self.opt.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop
        else:
            raise ValueError(f"Invalid optimiser {self.opt}")

        state_input, action_output = self._model_architecture()
        model = keras.Model(inputs=[state_input], outputs=[action_output], name=model_name)
        model.compile(optimizer=opt(learning_rate=self.learning_rate), loss=loss)

        return model

    @abc.abstractmethod
    def _model_architecture(self) -> Tuple[keras.layers.Layer, keras.layers.Layer]:
        """Define model construction function. Should return input layer and output layer."""
        pass

    def plot(self, model_name: str = 'model') -> None:
        keras.utils.plot_model(self.compile(model_name), to_file=f"{model_name}.png", show_shapes=True)
