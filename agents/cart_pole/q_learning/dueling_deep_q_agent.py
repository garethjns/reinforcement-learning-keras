from tensorflow import keras
from tensorflow.keras import backend as K

from agents.cart_pole.q_learning.deep_q_agent import DeepQAgent


class DuelingDeepQAgent(DeepQAgent):
    """Exactly the same as the DQN but with a slightly modified model architecture."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = 'DuelingDQNAgent'

    def _build_model_copy(self, model_name: str) -> keras.Model:
        input = keras.layers.Input(name='input', shape=self.env.observation_space.shape)
        fc1 = keras.layers.Dense(units=16, name='fc1', activation='relu')(input)
        fc2 = keras.layers.Dense(units=8, name='fc2', activation='relu')(fc1)

        # Separate layers for baseline value (1 node) and action advantages (n action nodes)
        v_layer = keras.layers.Dense(1, activation='linear')(fc2)
        a_layer = keras.layers.Dense(self.env.action_space.n, activation='linear')(fc2)

        # Function definition for lambda layer. Base value layer + actions value layer - mean action values
        def _merge_layer(layer_inputs: list):
            return layer_inputs[0] + layer_inputs[1] - K.mean(layer_inputs[1], axis=1, keepdims=True)

        output = keras.layers.Lambda(_merge_layer, output_shape=(self.env.action_space.n,),
                                     name="output")([v_layer, a_layer])

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model = keras.Model(inputs=[input], outputs=[output],
                            name=model_name)
        model.compile(opt, loss='mse')

        return model


if __name__ == "__main__":
    DuelingDeepQAgent.example()
