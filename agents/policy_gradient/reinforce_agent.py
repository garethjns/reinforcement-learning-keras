from dataclasses import dataclass
from typing import List, Tuple, Union, Dict, Any, Callable, Iterable

import numpy as np
from tensorflow.keras import backend as K

from agents.agent_base import AgentBase
from agents.components.helpers.env_builder import EnvBuilder
from agents.components.helpers.virtual_gpu import VirtualGPU
from agents.components.history.training_history import TrainingHistory
from enviroments.config_base import ConfigBase
from enviroments.model_base import ModelBase


class BaseModel(object):
    pass


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
    training_history: TrainingHistory
    model_architecture: ModelBase
    env_spec: str = "CartPole-v0"
    env_wrappers: Iterable[Callable] = ()
    name: str = 'REINFORCEAgent'
    alpha: float = 0.001
    gamma: float = 0.99
    learning_rate: float = 0.001

    _model_weights: Union[np.ndarray, None] = None

    def __post_init__(self) -> None:
        self.env_builder = EnvBuilder(self.env_spec, self.env_wrappers)

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

    def _build_model(self) -> None:
        """State -> model -> action probs"""
        self._model = self.model_architecture.compile(model_name='action_model')

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
        y = action_probs + self.alpha * dlogps

        # Train
        x = self.transform(states)
        self._model.train_on_batch(x, y)

    def _play_episode(self, max_episode_steps: int = 500,
                      training: bool = False, render: bool = True) -> Tuple[float, int]:
        self.env._max_episode_steps = max_episode_steps

        obs = self.env.reset()
        total_reward = 0

        for frame in range(max_episode_steps):
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

        return total_reward, frame

    def _after_episode_update(self) -> None:
        """Monte-Carlo update of policy model is updated (ie. after each full episode, or more)"""
        self.update_model()
        self.clear_memory()

    @classmethod
    def example(cls, config: ConfigBase, render: bool = True,
                n_episodes: int = 500, max_episode_steps: int = 500, update_every: int = 1) -> "ReinforceAgent":
        """Create, train, and save agent for a given config."""
        VirtualGPU(config.gpu_memory)
        config_dict = config.build()

        agent = cls(**config_dict)

        agent.train(verbose=True, render=render,
                    n_episodes=n_episodes, max_episode_steps=max_episode_steps, update_every=update_every)
        agent.save(f"{agent.name}_{config_dict['env_spec']}.pkl")

        return agent


if __name__ == "__main__":
    from enviroments.cart_pole.cart_pole_config import CartPoleConfig

    agent_cart_pole = ReinforceAgent.example(CartPoleConfig(agent_type='reinforce', plot_during_training=True))
