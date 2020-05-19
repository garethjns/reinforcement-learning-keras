import copy
from dataclasses import dataclass
from typing import Dict, Any, Union

import gym
import numpy as np
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from agents.agent_base import AgentBase
from agents.cart_pole.environment_processing.clipper import Clipper
from agents.cart_pole.q_learning.components.epsilon_greedy import EpsilonGreedy
from agents.cart_pole.q_learning.components.replay_buffer import ReplayBuffer
from agents.plotting.training_history import TrainingHistory


@dataclass
class DQNAgent(AgentBase):
    env_spec: str = "CartPole-v0"
    name: str = 'DQNAgent'
    eps: EpsilonGreedy = None
    gamma: float = 0.99
    plot_during_training: bool = True
    replay_buffer: ReplayBuffer = None
    replay_buffer_samples: int = 75

    _action_model_weights: Union[np.ndarray, None] = None
    _value_model_weights: Union[np.ndarray, None] = None

    def __post_init__(self) -> None:
        self.history = TrainingHistory(plotting_on=self.plot_during_training,
                                       plot_every=25,
                                       rolling_average=12,
                                       agent_name=self.name)

        if self.eps is None:
            # Prepare the default EpsilonGreedy sampler if one is not specified.
            self.eps = EpsilonGreedy(eps_initial=0.9,
                                     decay=0.002,
                                     eps_min=0.01)

        if self.replay_buffer is None:
            # Prepare the default ReplayBuffer if one is not specified.
            self.replay_buffer = ReplayBuffer(buffer_size=200)

        self._set_env()
        self._build_pp()
        self._build_model()

    def _build_model(self) -> None:
        """
        Prepare two of the same model.

        The action model is used to pick actions and the value model is used to predict value of Q(s', a). Action model
        weights are updated on every buffer sample + training step. The value model is never directly trained, but it's
        weights are updated to match the action model at the end of each episode.

        :return:
        """
        self._action_model = self._build_model_copy('action_model')
        self._value_model = self._build_model_copy('value_model')

        # If existing model weights have been passed at object instantiation, apply these. This is likely will only
        # be done when unpickling or when preparing to pickle this object.
        if self._action_model_weights is not None:
            self._action_model.set_weights(self._action_model_weights)
            self._action_model_weights = None
        if self._value_model_weights is not None:
            self._value_model.set_weights(self._value_model_weights)
            self._value_model_weights = None

    def _build_pp(self) -> None:
        """
        Create and fit the pre-processing pipeline.

        raw obs -> clip -> standard scaler
        """

        # Sample observations from environment to train scaler.
        obs = np.array([self.env.observation_space.sample() for _ in range(100000)])

        pipe = Pipeline([('clip', Clipper(lim=(-100, 100))),
                         ('ss', StandardScaler())])
        pipe.fit(obs)

        self.pp = pipe

    def _build_model_copy(self, model_name: str) -> keras.Model:
        """
        Prepare the neural network architecture for the action model (and it's less-often-updated value model copy).
        state -> NN -> [action value 1, action value 2]

        :param model_name: Model name.
        """

        state_input = keras.layers.Input(name='input', shape=self.env.observation_space.shape)
        fc1 = keras.layers.Dense(units=16, name='fc1', activation='relu')(state_input)
        fc2 = keras.layers.Dense(units=8, name='fc2', activation='relu')(fc1)
        output = keras.layers.Dense(units=self.env.action_space.n, name='output', activation=None)(fc2)

        opt = keras.optimizers.Adam(learning_rate=0.001)
        model = keras.Model(inputs=[state_input], outputs=[output],
                            name=model_name)
        model.compile(opt, loss='mse')

        return model

    def transform(self, s: np.ndarray) -> np.ndarray:
        """Run the processing pipeline on a single state."""
        if len(s.shape) == 1:
            s = s.reshape(1, -1)

        return self.pp.transform(s)

    def update_experience(self, s: np.ndarray, a: int, r: float, d: bool) -> None:
        """
        First the most recent step is added to the buffer.

        Note that s' isn't saved because there's no need. It'll be added next step. s' for any s is always index + 1 in
        the buffer.
        """

        # Add s, a, r, d to experience buffer
        self.replay_buffer.append((s, a, r, d))

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
        ss, aa, rr, dd, ss_ = self.replay_buffer.sample(self.replay_buffer_samples)

        # For each sample, calculate targets using Bellman eq and value/target network
        states1 = np.vstack(ss)
        states2 = np.vstack(ss_)
        y_now = self._value_model.predict(states1)
        y_future = self._value_model.predict(states2)
        y = []
        for i, (state, action, reward, done, state_) in enumerate(zip(ss, aa, rr, dd, ss_)):
            if done:
                # If done, reward is just this step. For cart-pole can only be done if agent has failed, so punish.
                g = - 10
            else:
                # Otherwise, it's the reward plus the predicted max value of next action
                g = reward + self.gamma * np.max(y_future[i, :])

            # Set non-acted actions to y_now preds and acted action to y_future pred
            y_ = y_now[i, :]
            y_[action] = g

            y.append(y_)

        # Fit action
        self._action_model.train_on_batch(states1, np.stack(y))

    def get_best_action(self, s: np.ndarray) -> np.ndarray:
        """
        Get best action(s) from model - the one with the highest predicted value.
        :param s: A single or multiple rows of state observations.
        :return: The selected action.
        """
        preds = self._action_model.predict(np.atleast_2d(s))

        return np.argmax(preds)

    def get_action(self, s: np.ndarray, training: bool = False) -> int:
        """
        Get an action using epsilon greedy.

        Epsilon decays every time a random action is chosen.

        :param s: The raw state observation.
        :param training: Bool to indicate whether or not to use this experience to update the model. If False, just
                         returns best action.
        :return: The selected action.
        """
        action = self.eps.select(greedy_option=lambda: self.get_best_action(s),
                                 random_option=lambda: self.env.action_space.sample(),
                                 training=training)

        return action

    def update_value_model(self):
        """
        Update the value model with the weights of the action model (which is updated each step).

        The value model is updated less often to aid stability.
        """
        self._value_model.set_weights(self._action_model.get_weights())

    def play_episode(self, max_episode_steps: int = 500,
                     training: bool = False, render: bool = True) -> float:
        """
        Play a single episode and return the total reward.

        :param max_episode_steps: Max steps before stopping, overrides any time limit set by Gym.
        :param training: Bool to indicate whether or not to use this experience to update the model.
        :param render: Bool to indicate whether or not to call env.render() each training step.
        :return: The total real reward for the episode.
        """
        obs = self.env.reset()
        total_reward = 0
        for _ in range(max_episode_steps):
            action = self.get_action(obs, training=training)
            prev_obs = obs
            obs, reward, done, info = self.env.step(action)
            total_reward += reward

            if render:
                self.env.render()

            if training:
                self.update_experience(s=prev_obs, a=action, r=reward, d=done)
                self.update_model()

            if done:
                break

        return total_reward

    def train(self, n_episodes: int = 100, max_episode_steps: int = 500,
              verbose: bool = True, render: bool = True) -> None:
        """
        Run the training loop. It's the same as the linear agent version, + the value model update.

        :param n_episodes: Number of episodes to run.
        :param max_episode_steps: Max steps before stopping, overrides any time limit set by Gym.
        :param verbose:  If verbose, use tqdm and print last episode score for feedback during training.
        :param render: Bool to indicate whether or not to call env.render() each training step.
        """
        self.env._max_episode_steps = max_episode_steps
        self._set_tqdm(verbose)

        for _ in self._tqdm(range(n_episodes)):
            total_reward = self.play_episode(max_episode_steps,
                                             training=True, render=render)
            # Value model is at the end of each episode
            self.update_value_model()

            self._update_history(total_reward, verbose)

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare object for pickling.

        The env and Keras model objects can't be pickled with pickle. Remove these, but put them back afterwards
        so this object is effectively unchanged.

        Build models will do the same thing here as when the model is unpicked. With the _action_model_weights and
        _value_model_weights attributes/kwargs set, it'll make the model architecture and set these weights.
        """

        # Remove things
        for m in ['_action_model', '_value_model']:
            self.__dict__[f"{m}_weights"] = self.__dict__[m].get_weights()
            self.__dict__[m] = None
        self.__dict__["env"] = None

        # Get object spec to pickle
        object_state_dict = copy.deepcopy(self.__dict__)

        # Put this object back how it was
        self._set_env()
        self._build_model()

        return object_state_dict


if __name__ == "__main__":

    try:
        tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=1024)])
    except (IndexError, AttributeError):
        # Either no GPU available or virtual device config failed for some reason. Can probably continue.
        pass

    agent = DQNAgent("CartPole-v0")
    agent.train(verbose=True, render=False, n_episodes=5)

    agent.save('asd')
    agent_2 = DQNAgent.load('asd')

