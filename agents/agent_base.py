import abc
import pickle
from typing import Any, Callable, Union

import gym
from tqdm import tqdm

from agents.plotting.training_history import TrainingHistory


class AgentBase(abc.ABC):
    history: TrainingHistory
    env_spec: str
    name: str
    _env: Union[None, gym.Env]
    _tqdm: Callable

    def check_ready(self) -> None:
        """
        Check the model is ready to use.

        Default implementation:
         - Check _env is set (most models?)
        Example of other model specific steps that might need doing:
         - For Keras models, check model is ready, for example if it needs recompiling after loading.
        """
        self._set_env()

    def unready(self) -> None:
        """Remove anything that causes issues with pickling, such as the env or keras models."""
        self._env = None

    @classmethod
    def set_tf(cls, gpu_memory_limit: int = 512, gpu_device_id: int = 0) -> None:
        """
        Helper function for training on tf. Reduces GPU memory footprint for keras/tf models.

        Creates a virtual device on the request GPU with limited memory. Will fail gracefully if GPU isn't available.

        :param gpu_memory_limit: Max memory in MB for  virtual device. Setting LOWER than total available this can help
                                 avoid out of memory errors on some set ups when TF tries to allocate too much memory
                                 (seems to be a bug).
        :param gpu_device_id: Integer device identifier for the real GPU the virtual GPU should use.
        """
        import tensorflow as tf

        try:
            # Handle running on GPU: If available, reduce memory commitment to avoid over-committing error in 2.2.0 and
            # for also for convenience.
            tf.config.experimental.set_virtual_device_configuration(
                tf.config.experimental.list_physical_devices('GPU')[gpu_device_id],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=gpu_memory_limit)])
        except (IndexError, AttributeError):
            # Either no GPU available or virtual device config failed for some reason. Can probably continue.
            pass

    @classmethod
    def example(cls):
        """Optional example function using this agent."""
        raise NotImplementedError

    def _set_env(self):
        """Create a new env object from the requested spec."""
        self._env = gym.make(self.env_spec)

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
    def update_model(self, *args, **kwargs) -> None:
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
