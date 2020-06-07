from typing import List, Callable

import numpy as np
from tensorflow import keras

from agents.cart_pole.q_learning.deep_q_agent import DeepQAgent as CartDeepQAgent
from agents.pong.pong_environment_templates import WRAPPERS_DIFF, WRAPPERS_STACK
from agents.virtual_gpu import VirtualGPU


class DeepQAgent(CartDeepQAgent):
    """
    DeepQAgent set up for the Pong environment.

    frame_depth controls two available modes for the FrameBufferWrapper. This wrapper buffers frames before (before the agent sees them) and can either return a sequence of frames or the diff between the last two frames.
     - frame_depth = 1 uses 'diff' mode of FrameBufferWrapper to return diff of last two frames in shape (84, 84, 1)
     - frame_depth > 1 returns sequence of n frames as channels shape (84, 84, n)
    """
    env_spec: str = "Pong-v0"
    learning_rate: float = 0.1
    frame_depth: int = 1

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

    def transform(self, s: np.ndarray) -> np.ndarray:
        """Run the any pre-processing on raw state, if used."""
        if len(s.shape) == 3:
            s = np.expand_dims(s, axis=0)

        return s

    def _build_model_copy(self, model_name: str):
        frame_input = keras.layers.Input(name='input', shape=(84, 84, self.frame_depth))
        conv1 = keras.layers.Conv2D(16, kernel_size=(9, 9),
                                    strides=(2, 2),
                                    name='conv1',
                                    padding='same',
                                    activation='relu')(frame_input)
        conv2 = keras.layers.Conv2D(32, kernel_size=(6, 6),
                                    strides=(1, 1),
                                    name='conv2',
                                    padding='same',
                                    activation='relu')(conv1)
        conv3 = keras.layers.Conv2D(64, kernel_size=(3, 3),
                                    strides=(1, 1),
                                    name='conv2',
                                    padding='same',
                                    activation='relu')(conv2)
        flatten = keras.layers.Flatten(name='flatten')(conv3)
        fc1 = keras.layers.Dense(units=300, name='fc1', activation='relu')(flatten)
        fc2 = keras.layers.Dense(units=64, name='fc2', activation='relu')(fc1)
        action_value_output = keras.layers.Dense(units=self.env.action_space.n, name='output',
                                                 activation=None)(fc2)

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model = keras.Model(inputs=[frame_input], outputs=[action_value_output],
                            name=model_name)
        model.compile(opt, loss='mse')

        return model

    def update_model(self) -> None:
        """
        Sample a batch from the replay buffer, calculate targets using value model, and train action model.

        If the buffer is below its minimum size, no training is done.

        If the buffer has reached its minimum size, a training batch from the replay buffer and the action model is
        updated.

        This update samples random (s, a, r, s') sets from the buffer and calculates the discounted reward for each set.
        The value of the actions at states s and s' are predicted from the value model. The action model is updated
        using these value predictions as the targets. The value of performed action is updated with the discounted
        reward (using its value prediction at s'). ie. x=s, y=[action value 1, action value 2].

        The predictions from the value model s, s', and the update of the action model is done in batch before and
        after the loop. The loop then iterates over the rows. Note that an alternative is doing the prediction and
        fit calls on singles rows in the loop. This would be very inefficient, especially if using a GPU.
        """

        # If buffer isn't full, don't train
        if not self.replay_buffer.full:
            return

        # Else sample batch from buffer
        ss, aa, rr, dd, ss_ = self.replay_buffer.sample_batch(self.replay_buffer_samples)

        # For each sample, calculate targets using Bellman eq and value/target network
        states1 = np.stack(ss)
        states2 = np.stack(ss_)
        y_now = self._value_model.predict(states1)
        y_future = self._value_model.predict(states2)
        y = []
        for i, (action, reward, done) in enumerate(zip(aa, rr, dd)):
            # Set non-acted actions to y_now preds and acted action to discounted reward
            y_ = y_now[i, :]
            y_[action] = self._get_reward(reward, y_future[i, :], done)
            y.append(y_)

        # Fit action
        self._action_model.train_on_batch(states1, np.stack(y))

    def get_best_action(self, s: np.ndarray) -> np.ndarray:
        """
        Get best action(s) from model - the one with the highest predicted value.
        :param s: A single or multiple rows of state observations.
        :return: The selected action.
        """
        preds = self._action_model.predict(self.transform(s))

        return np.argmax(preds)

    @classmethod
    def example(cls, n_episodes: int = 5000, render: bool = True) -> "DeepQAgent":
        """Run a quick example with n_episodes and otherwise default settings."""
        VirtualGPU(512)
        agent = cls("Pong-v0")
        agent.train(verbose=True, render=render,
                    max_episode_steps=2000,
                    n_episodes=n_episodes,
                    checkpoint_every=False)

        return agent


if __name__ == "__main__":
    agent_ = DeepQAgent.example(render=True)
