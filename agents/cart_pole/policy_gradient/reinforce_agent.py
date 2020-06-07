from dataclasses import dataclass
from typing import List, Tuple, Union, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from agents.agent_base import AgentBase
from agents.plotting.training_history import TrainingHistory
from agents.agent_helpers.virtual_gpu import VirtualGPU


@dataclass
class ReinforceAgent(AgentBase):
    """
    Uses a simple replay buffer.

    Has 2 components:
      - _current_ : List of steps being collected for current episode
      - _buffer_ : Dict containing backlog of completed episodes not yet used for training model

    At the end of an episode, the current episode is moved to the backlog. This is cleared after updating model,
    which can occur less often.
    """
    env_spec: str = "CartPole-v0"
    name: str = 'REINFORCEAgent'
    lr: float = 0.001
    gamma: float = 0.99
    plot_during_training: bool = True
    learning_rate: float = 0.001

    _model_weights: Union[np.ndarray, None] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.history = TrainingHistory(plotting_on=self.plot_during_training,
                                       plot_every=25,
                                       rolling_average=12,
                                       agent_name=self.name)

        # Keep track of number of trained episodes, only used for IDing episodes in buffer.
        self._ep_tracker: int = -1

        self._build_model()
        self.clear_memory()

    def __getstate__(self) -> Dict[str, Any]:
        return self._pickle_compatible_getstate()

    def unready(self) -> None:
        super().unready()
        if self._model is not None:
            self._model_weights = self._model.get_weights()
            self._model = None

    def check_ready(self) -> None:
        super().check_ready()
        if self._model is None:
            self._build_model()

    def _clear_current_episode(self) -> None:
        """Clear buffer for current episode."""
        self._current_states: List[np.ndarray] = []
        self._current_action_probs: List[np.ndarray] = []
        self._current_actions: List[int] = []
        self._current_rewards: List[float] = []

    def _clear_buffer_backlog(self) -> None:
        """Clear backlog of collected episodes not yet trained on."""
        self._buffer_states: Dict[int, List[np.ndarray]] = {}
        self._buffer_action_probs: Dict[int, List[np.ndarray]] = {}
        self._buffer_actions: Dict[int, np.ndarray] = {}
        self._buffer_rewards: Dict[int, np.ndarray] = {}
        self._buffer_discounted_rewards: Dict[int, np.ndarray] = {}

    def _move_current_episode_to_backlog(self, episode: int):
        """Move current episode to backlog, calc discounted rewards, and clear."""
        self._buffer_states[episode] = self._current_states
        self._buffer_action_probs[episode] = self._current_action_probs
        self._buffer_actions[episode] = np.array(self._current_actions)
        self._buffer_rewards[episode] = np.array(self._current_rewards)
        self._buffer_discounted_rewards[episode] = self._calc_discounted_rewards(self._current_rewards)
        self._clear_current_episode()

    def clear_memory(self) -> None:
        """Clear current episode and backlog buffers."""
        self._clear_current_episode()
        self._clear_buffer_backlog()

    @staticmethod
    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return - tf.reduce_sum(y_true * tf.math.log(y_pred),
                               axis=1)

    def _build_model(self) -> None:
        """
        State -> model -> action probs
        """
        state_input = keras.layers.Input(shape=self.env.observation_space.shape)
        fc1 = keras.layers.Dense(24, activation='relu')(state_input)
        fc2 = keras.layers.Dense(12, activation='relu')(fc1)
        action_output = keras.layers.Dense(self.env.action_space.n, activation='softmax')(fc2)

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model = keras.Model(state_input, action_output)
        model.compile(optimizer=opt, loss=self._loss)

        self._model = model

        # If existing model weights have been passed at object instantiation, apply these. This is likely will only
        # be done when unpickling or when preparing to pickle this object.
        if self._model_weights is not None:
            self._model.set_weights(self._model_weights)
            self._model_weights = None

    def transform(self, s: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
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
        return actions_probs, np.random.choice(range(self.env.action_space.n),
                                               p=actions_probs)

    def update_experience(self, s: np.ndarray, a: int, r: float, a_p: np.ndarray) -> None:
        """
        Add step of experience to the buffer.

        :param s: State
        :param a: Action
        :param r: Reward
        :param a_p: Action probabilities
        """
        self._current_states.append(s)
        self._current_action_probs.append(a_p)
        self._current_actions.append(a)
        self._current_rewards.append(r)

    def _calc_discounted_rewards(self, rr: List[float]) -> np.ndarray:
        """Calculate discounted rewards for a whole episode and normalise."""

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

    @staticmethod
    def _flatten_list(nested_list: List[List[Any]]) -> List[Any]:
        return [item for sublist in nested_list for item in sublist]

    def update_model(self) -> None:
        # Stack all available episodes

        states = np.concatenate(list(self._buffer_states.values()))
        disc_rewards = np.concatenate(list(self._buffer_discounted_rewards.values()))
        actions = np.concatenate(list(self._buffer_actions.values())).reshape(-1, 1)
        action_probs = np.vstack(list(self._buffer_action_probs.values()))

        # One hot actions
        actions_oh = K.one_hot(actions,
                               num_classes=self.env.action_space.n)

        # Calculate prob updates
        dlogps = (actions_oh - action_probs) * disc_rewards
        y = action_probs + self.lr * dlogps

        # Train
        x = self.transform(states)
        self._model.train_on_batch(x, y)

    def play_episode(self, max_episode_steps: int = 500,
                     training: bool = False, render: bool = True) -> float:
        self.env._max_episode_steps = max_episode_steps

        obs = self.env.reset()
        total_reward = 0

        for _ in range(max_episode_steps):
            action_probs, action = self.get_action(obs)
            prev_obs = obs
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward

            if render:
                self.env.render()

            if training:
                self.update_experience(s=prev_obs, a=action, r=reward, a_p=action_probs)

            if done:
                break

        if training:
            # Only keep episode buffer if actually training
            self._ep_tracker += 1
            self._move_current_episode_to_backlog(self._ep_tracker)

        return total_reward

    def _after_episode_update(self) -> None:
        """Monte-Carlo update of policy model is updated (ie. after each full episode, or more)"""
        self.update_model()
        self.clear_memory()

    @classmethod
    def example(cls, n_episodes: int = 1000, render: bool = True) -> "ReinforceAgent":
        """Run a quick example with n_episodes and otherwise default settings."""
        VirtualGPU(256)
        agent = cls("CartPole-v0")
        agent.train(verbose=True, render=render,
                    update_every=3,
                    n_episodes=n_episodes)

        return agent


if __name__ == "__main__":
    ReinforceAgent.example(render=False)
