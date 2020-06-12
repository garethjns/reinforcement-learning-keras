from typing import List, Callable, Union

import numpy as np
from tensorflow import keras

from agents.cart_pole.policy_gradient.reinforce_agent import ReinforceAgent as CartReinforceAgent
from agents.history.training_history import TrainingHistory
from agents.pong.pong_environment_templates import WRAPPERS_DIFF, WRAPPERS_STACK
from agents.virtual_gpu import VirtualGPU


class ReinforceAgent(CartReinforceAgent):
    env_spec: str = "Pong-v0"
    env_wrappers: List[Callable] = None
    frame_depth: int = 1

    def __post_init__(self) -> None:
        self.env_spec: str = "Pong-v0"
        self.history = TrainingHistory(plotting_on=self.plot_during_training,
                                       plot_every=25,
                                       rolling_average=12,
                                       agent_name=self.name)

        self._set_env()
        self._build_model()
        self.clear_memory()

    @property
    def env_wrappers(self) -> List[Callable]:
        """
        Wrappers to add to environment when built. Order is [Applied first (inner), ...,  applied last (outer)]

        :return: List of wrappers. Can include partial functions.

        Here frame_buffer param sets diff or sequence mode of FrameBufferWrapper.
        """
        return WRAPPERS_DIFF if self.frame_depth == 1 else WRAPPERS_STACK

    def _build_pp(self) -> None:
        """Prepare pre-processor for the raw state, if needed."""
        pass

    def transform(self, s: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """Run the any pre-processing on raw state, if used."""
        if not isinstance(s, np.ndarray):
            s = np.stack(s)

        if len(s.shape) == 3:
            s = np.expand_dims(s, axis=0)

        return s

    def _build_model(self):
        frame_input = keras.layers.Input(name='input', shape=(84, 84, 1))
        conv1 = keras.layers.Conv2D(32, kernel_size=(6, 6),
                                    strides=(2, 2),
                                    name='conv1',
                                    padding='same',
                                    activation='relu')(frame_input)
        conv2 = keras.layers.Conv2D(16, kernel_size=(6, 6),
                                    strides=(2, 2),
                                    name='conv2',
                                    padding='same',
                                    activation='relu')(conv1)

        flatten = keras.layers.Flatten(name='flatten')(conv2)
        fc1 = keras.layers.Dense(units=128, name='fc1', activation='relu')(flatten)
        fc2 = keras.layers.Dense(units=64, name='fc2', activation='relu')(fc1)
        action_output = keras.layers.Dense(self.env.action_space.n, activation='softmax')(fc2)

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model = keras.Model(inputs=[frame_input], outputs=[action_output])
        model.compile(optimizer=opt, loss=self._loss)

        self._model = model

        keras.utils.plot_model(model, to_file="pong_reinforce.png", show_shapes=True)

        # If existing model weights have been passed at object instantiation, apply these. This is likely will only
        # be done when unpickling or when preparing to pickle this object.
        if self._model_weights is not None:
            self._model.set_weights(self._model_weights)
            self._model_weights = None

        return model

    @classmethod
    def example(cls, n_episodes: int = 500, render: bool = True) -> "ReinforceAgent":
        """Run a quick example with n_episodes and otherwise default settings."""
        VirtualGPU(1024)
        agent = cls("Pong-v0")
        agent.train(verbose=True, render=render,
                    max_episode_steps=1000,
                    n_episodes=n_episodes)

        return agent


if __name__ == "__main__":
    agent = ReinforceAgent.example(render=True)
    agent.save("test_pong_dqn.pkl")
