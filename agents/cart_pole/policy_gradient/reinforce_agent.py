from dataclasses import dataclass
from typing import List, Tuple, Union, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from agents.agent_base import AgentBase
from agents.plotting.training_history import TrainingHistory


@dataclass
class ReinforceAgent(AgentBase):
    env_spec: str = "CartPole-v0"
    name: str = 'REINFORCEAgent'
    lr: float = 0.001
    gamma: float = 0.99
    plot_during_training: bool = True
    learning_rate: float = 0.001

    _model_weights: Union[np.ndarray, None] = None

    def __post_init__(self) -> None:
        self.history = TrainingHistory(plotting_on=self.plot_during_training,
                                       plot_every=25,
                                       rolling_average=12,
                                       agent_name=self.name)

        self._set_env()
        self._build_model()
        self.clear_memory()

    def __getstate__(self) -> Dict[str, Any]:
        return self._pickle_compatible_getstate()

    def unready(self) -> None:
        super().unready()
        if self._model is not None:
            self._weights = self._model.get_weights()
            self._model = None

    def check_ready(self) -> None:
        super().check_ready()
        if self._model is None:
            self._build_model()

    def clear_memory(self) -> None:
        self._states: List[np.ndarray] = []
        self._action_probs: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []

    @staticmethod
    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return - tf.reduce_sum(y_true * tf.math.log(y_pred),
                               axis=1)

    def _build_model(self) -> None:
        """
        State -> model -> action probs
        """
        state_input = keras.layers.Input(shape=self._env.observation_space.shape)
        fc1 = keras.layers.Dense(24, activation='relu')(state_input)
        fc2 = keras.layers.Dense(12, activation='relu')(fc1)
        action_output = keras.layers.Dense(self._env.action_space.n, activation='softmax')(fc2)

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model = keras.Model(state_input, action_output)
        model.compile(optimizer=opt, loss=self._loss)

        self._model = model

        # If existing model weights have been passed at object instantiation, apply these. This is likely will only
        # be done when unpickling or when preparing to pickle this object.
        if self._model_weights is not None:
            self._model.set_weights(self._model_weights)
            self._model_weights = None

    def transform(self, s: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """No transforming of state here, just stacking and dimension checking.
        """
        if not isinstance(s, np.ndarray):
            s = np.vstack(s)

        return np.atleast_2d(s)

    def get_action(self, s: np.ndarray, training=None) -> Tuple[np.ndarray, int]:
        """
        Use the current policy to select an action from a single state observation.

        Sample actions using the probabilities provided by the action model.
        """
        actions_probs = self._model.predict(self.transform(s)).squeeze()
        return actions_probs, np.random.choice(range(self._env.action_space.n),
                                               p=actions_probs)

    def update_experience(self, s: np.ndarray, a: int, r: float, a_p: np.ndarray) -> None:
        self._states.append(s)
        self._action_probs.append(a_p)
        self._actions.append(a)
        self._rewards.append(r)

    def calc_discounted_rewards(self, rr) -> np.ndarray:
        """Calculate discounted rewards for whole episode and normalise."""

        # Full episode returns
        disc_rr = np.zeros_like(rr)
        cumulative_reward = 0
        for t in reversed(range(0, disc_rr.size)):
            cumulative_reward = cumulative_reward * self.gamma + rr[t]
            disc_rr[t] = cumulative_reward

        # Normalise
        disc_rr_mean = np.mean(disc_rr)
        disc_rr_std = np.std(disc_rr) + 1e-9
        disc_rr_norm = (disc_rr - disc_rr_mean) / disc_rr_std

        return np.vstack(disc_rr_norm)

    def update_model(self) -> None:
        # Calc discounted rewards for last episode in buffer
        disc_rr_norm = self.calc_discounted_rewards(self._rewards)

        # One hot actions
        actions_oh = K.one_hot(self._actions,
                               num_classes=self._env.action_space.n)

        # Calculate prob updates
        dlogps = (actions_oh - np.vstack(self._action_probs)) * disc_rr_norm
        y = np.vstack(self._action_probs) + self.lr * dlogps

        # Train
        x = self.transform(self._states)
        self._model.train_on_batch(x, y)

    def play_episode(self, max_episode_steps: int = 500,
                     training: bool = False, render: bool = True) -> float:
        self.clear_memory()

        obs = self._env.reset()
        total_reward = 0
        for _ in range(max_episode_steps):
            action_probs, action = self.get_action(obs)
            prev_obs = obs
            obs, reward, done, _ = self._env.step(action)
            total_reward += reward

            if render:
                self._env.render()

            if training:
                self.update_experience(prev_obs, action_probs, action, reward)

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
        self._set_env()
        self._set_tqdm(verbose)

        for _ in self._tqdm(range(n_episodes)):
            total_reward = self.play_episode(max_episode_steps,
                                             training=True, render=render)
            # Monte-Carlo update of policy model is updated (ie. after each full episode)
            self.update_model()

            self._update_history(total_reward, verbose)

    @classmethod
    def example(cls, n_episodes: int = 500, render: bool = True) -> "ReinforceAgent":
        """Run a quick example with n_episodes and otherwise default settings."""
        cls.set_tf(256)
        agent = cls("CartPole-v0")
        agent.train(verbose=True, render=render,
                    n_episodes=n_episodes)

        return agent


if __name__ == "__main__":
    ReinforceAgent.example()
