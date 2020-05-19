import abc
import pickle
from typing import Any, Callable

from tqdm import tqdm

from agents.plotting.training_history import TrainingHistory
import gym


class AgentBase(abc.ABC):
    history: TrainingHistory
    env_spec: str
    env: gym.Env
    _tqdm: Callable

    def _set_env(self):
        """Create a new env object from the requested spec."""
        self.env = gym.make(self.env_spec)

    def _build_pp(self):
        """Prepare pre-processor for the raw state, if needed."""
        pass

    def transform(self, s: Any) -> Any:
        """Run the any pre-processing on raw state, if used."""
        return s

    def update_experience(self, s: Any, a: int, r: float, d: bool) -> None:
        """Remember an experience, if used by agent."""
        pass

    @abc.abstractmethod
    def _build_model(self) -> None:
        """Prepare the model(s) the agent will use."""
        pass

    @abc.abstractmethod
    def update_model(self) -> None:
        """Update the agents model(s)."""
        pass

    @abc.abstractmethod
    def get_action(self, s: Any, **kwargs) -> int:
        """
        Given state s, get an action from the agent.

        May include other kwargs if needed - for example, a training flag for methods using epsilon greedy.
        """
        pass

    @abc.abstractmethod
    def play_episode(self, max_episode_steps: int = 500,
                     training: bool = False, render: bool = True) -> float:
        """
        Play a single episode with the agent (run multiple steps).

        :param max_episode_steps: Max steps before stopping, overrides any time limit set by Gym.
        :param training: Bool to indicate whether or not to use this experience to update the model.
        :param render: Bool to indicate whether or not to call env.render() each training step.
        :return: The total real reward for the episode.
        """
        pass

    @abc.abstractmethod
    def train(self, n_episodes: int = 100, max_episode_steps: int = 500,
              verbose: bool = True, render: bool = True) -> None:
        """
        Defines the training loop for the agent.

        :param n_episodes: Number of episodes to run.
        :param max_episode_steps: Max steps before stopping, overrides any time limit set by Gym.
        :param verbose:  If verbose, use tqdm and print last episode score for feedback during training.
        :param render: Bool to indicate whether or not to call env.render() each training step.
        """
        pass

    def _set_tqdm(self, verbose: bool = False) -> None:
        """Turn tqdm on or of depending on verbosity setting."""
        _tqdm = tqdm if verbose else lambda x: x

        self._tqdm = _tqdm

    def _update_history(self, total_reward: float, verbose: bool = True) -> None:
        """
        Add an episodes reward to history and maybe plot depending on history settings.

        :param total_reward: Reward from last episode to append to history.
        :param verbose: If verbose, print the last episode and run the history plot. The history plot will display
                        depending on it's own settings. Verbose = False will turn it off totally.
        """
        self.history.append(total_reward)

        if verbose:
            print(total_reward)
            self.history.training_plot()

    def save(self, fn: str) -> None:
        """Apparently compiled Keras model can be pickled now?"""
        pickle.dump(self, open(fn, 'wb'))

    @classmethod
    def load(cls, fn: str) -> "AgentBase":
        """Eat pickle"""
        return pickle.load(open(fn, 'rb'))
