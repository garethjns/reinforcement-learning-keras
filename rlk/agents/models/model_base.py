import abc
from dataclasses import dataclass
from typing import Tuple, Union, Callable, List

from tensorflow import keras
from tensorflow.keras import backend as K


@dataclass
class ModelBase(abc.ABC):
    """
    :param observation_shape: Tuple specifying input shape.
    :param unit_scale: Multiplier for all units in FC layers in network. Default 1 = 16 units for first layer,
                        8 for second.
    :param opt: Keras optimiser to use. Should be string. This is to avoid storing TF/Keras objects here.
    :param learning_rate: Learning rate for optimiser.
    :param n_actions: Int specifying number of outputs.
    :param output_type: Type of output to add to the nextwork, 'q' or 'ac'. Note appropriate loss should also be set.
                         - 'q' - for q-learning, adds action value output (optionally dueling)
                         - 'ac' - for actor-critic, adds output for action and another for critic value
    :param output_activation: Activation function for output. Eg. None for value estimation (off-policy methods) or
                              'softmax' for action probabilities (on-policy methods).
    :param dueling: Use a dueling architecture on the model output. Not used if output type = 'ac'.
    """
    observation_shape: Tuple[int, ...]
    n_actions: int
    output_activation: Union[None, str] = None
    output_type: str = 'q'
    unit_scale: int = 1
    learning_rate: float = 0.0001
    opt: str = 'Adam'
    dueling: bool = False
    hidden_layer_activations: str = 'relu'

    def compile(self, model_name: str = 'model', loss: Union[str, Callable] = 'mse', **kwargs) -> keras.Model:
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

        state_inputs, action_outputs = self._model_architecture()
        model = keras.Model(inputs=state_inputs, outputs=action_outputs, name=model_name)
        model.compile(optimizer=opt(learning_rate=self.learning_rate), loss=loss, **kwargs)

        return model

    @abc.abstractmethod
    def _model_architecture(self) -> Tuple[List[keras.layers.Layer], List[keras.layers.Layer]]:
        """Define model construction function. Should return input layer and output layer."""
        pass

    def _add_output(self, input_layer: keras.layers.Layer) -> List[keras.layers.Layer]:
        if self.output_type == 'q':
            return [self._add_q_output(input_layer)]
        else:
            return self._add_ac_output(input_layer)

    def _add_ac_output(self, input_layer: keras.layers.Layer) -> List[keras.layers.Layer]:
        actor_output = keras.layers.Dense(self.n_actions, name='actor_output', activation='softmax')(input_layer)
        critic_output = keras.layers.Dense(1, activation=None, name='critic_output')(input_layer)

        return [actor_output, critic_output]

    def _add_q_output(self, input_layer: keras.layers.Layer) -> keras.layers.Layer:
        """Add the model output - either dueling or not."""
        if self.dueling:
            # Separate layers for baseline value (1 node) and action advantages (n action nodes)
            v_layer = keras.layers.Dense(1, activation='linear')(input_layer)
            a_layer = keras.layers.Dense(self.n_actions, activation='linear')(input_layer)

            # Function definition for lambda layer. Base value layer + actions value layer - mean action values
            def _merge_layer(layer_inputs: list):
                return layer_inputs[0] + layer_inputs[1] - K.mean(layer_inputs[1], axis=1, keepdims=True)

            action_output = keras.layers.Lambda(_merge_layer, output_shape=(self.n_actions,),
                                                name="output")([v_layer, a_layer])
        else:
            # Make a non-dueling output
            action_output = keras.layers.Dense(units=self.n_actions, name='output',
                                               activation=self.output_activation)(input_layer)

        return action_output

    def plot(self, model_name: str = 'model') -> None:
        keras.utils.plot_model(self.compile(model_name), to_file=f"{model_name}.png", show_shapes=True)
