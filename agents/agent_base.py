import abc
import copy
import pickle
from functools import reduce
from typing import Any, Callable, Union, Dict, List

import gym
import numpy as np
from tqdm import tqdm

from agents.plotting.training_history import TrainingHistory


class AgentBase(abc.ABC):
    history: TrainingHistory
    env_spec: str
    name: str
    gamma: float

    _env: Union[None, gym.Env] = None
    _tqdm: Callable

    def _pickle_compatible_getstate(self) -> Dict[str, Any]:
        """
        Prepare agent with a keras model object for pickling.

        Calls .unready to prepare this object for pickling, and .check_ready to put it back how it was after pickling.
        In addition to what's defined in the unready method, it avoids trying to copy the env. We can't copy this,
        but we also don't want to make a new one (like we can with the models). This would cause a new render window
        per episode...

        The default unready method in AgentBase does nothing. The GPU models should modify .unready and .check_ready to
        handle complied Keras, which also can't be pickled.

        Object that need to use this should implement their own __getstate__:
        def __getstate__(self) -> Dict[str, Any]:
            return self._pickle_compatible_getstate()
        It's not implemented in AgentBase as the standard __getstate__ is required by the deepcopy below.
        """

        # Remove things
        self.unready()

        # Get object spec to pickle, everything left except env
        object_state_dict = copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in ["_env", "env"]})

        # Put this object back how it was
        self.check_ready()

        return object_state_dict

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
        """Remove anything that causes issues with pickling, such as keras models."""
        pass

    @classmethod
    def example(cls) -> "AgentBase":
        """Optional example function using this agent."""
        raise NotImplementedError

    @property
    def env_wrappers(self) -> List[Callable]:
        return []

    def _set_env(self, env: Union[None, gym.Env] = None) -> None:
        """
        Create a new env object from the spec, or set a new one.

        Can specify a new env, this is useful, for example, to add a Monitor wrapper.
        """

        if env is not None:
            self._env = env

        if self._env is None:
            # Make the gym environment and apply the wrappers one by one
            self._env = reduce(lambda inner_env, wrapper: wrapper(inner_env),
                               self.env_wrappers,
                               gym.make(self.env_spec))

    @property
    def env(self) -> gym.Env:
        """Use to access env, if not ready also makes it ready."""
        self._set_env()

        return self._env

    def _build_pp(self) -> None:
        """Prepare pre-processor for the raw state, if needed."""
        pass

    def transform(self, s: Any) -> Any:
        """Run the any pre-processing on raw state, if used."""
        return s

    def update_experience(self, *args) -> None:
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

    @staticmethod
    def _final_reward(reward: float) -> float:
        """
        Use this to define reward for end of an episode. Default is just the reward for this step.

        For example:
          - cart pole: Something negative as end is always bad (assuming not a timeout)
          - Pong: Perhaps not negative - could win!
        """
        return reward

    def _discounted_reward(self, reward: float, estimated_future_action_rewards: np.ndarray) -> float:
        """Use this to define the discounted reward for unfinished episodes, default is 1 step TD."""
        return reward + self.gamma * np.max(estimated_future_action_rewards)

    def _get_reward(self, reward: float, estimated_future_action_rewards: np.ndarray, done: bool) -> float:
        """
        Calculate discounted reward for a single step.

        :param reward: Last real reward.
        :param estimated_future_action_rewards: Estimated future values of actions taken on next step.
        :param done: Flag indicating if this is the last step on an episode.
        :return: Reward.
        """

        if done:
            # If done, reward is just this step. Can finish because agent has won or lost.
            return self._final_reward(reward)
        else:
            # Otherwise, it's the reward plus the predicted max value of next action
            return self._discounted_reward(reward, estimated_future_action_rewards)

    @abc.abstractmethod
    def get_action(self, s: Any, **kwargs) -> int:
        """
        Given state s, get an action from the agent.

        May include other kwargs if needed - for example, a training flag for methods using epsilon greedy.
        """
        pass

    @abc.abstractmethod
    def play_episode(self, max_episode_steps: int = 500, training: bool = False, render: bool = True) -> float:
        """
        Play a single episode with the agent (run multiple steps).

        :param max_episode_steps: Max steps before stopping, overrides any time limit set by Gym.
        :param training: Bool to indicate whether or not to use this experience to update the model.
        :param render: Bool to indicate whether or not to call env.render() each training step.
        :return: The total real reward for the episode.
        """
        pass

    def train(self, n_episodes: int = 10000, max_episode_steps: int = 500, verbose: bool = True, render: bool = True,
              checkpoint_every: Union[bool, int] = 10, update_every: Union[bool, int] = 1) -> None:
        """
        Run the default training loop

        :param n_episodes: Number of episodes to run.
        :param max_episode_steps: Max steps before stopping, overrides any time limit set by Gym.
        :param verbose:  If verbose, use tqdm and print last episode score for feedback during training.
        :param render: Bool to indicate whether or not to call env.render() each training step.
        :param checkpoint_every: Save the model every n steps while training. Set to 0 or false to turn off.
        :param update_every: Run the _after_episode_update() step every n episodes.
        """
        self._set_tqdm(verbose)

        for ep in self._tqdm(range(n_episodes)):
            total_reward = self.play_episode(max_episode_steps,
                                             training=True,
                                             render=render)
            self._update_history(total_reward, verbose)

            if (update_every > 0) and not (ep % update_every):
                # Run the after-episode update step
                self._after_episode_update()

            if (checkpoint_every > 0) and not (ep % checkpoint_every):
                self.save(f"{self.name}_{self.env_spec}_checkpoint.pkl")

    def _after_episode_update(self) -> None:
        """
        Run an update step after an episode completes.

        In the default implementation of .train, update_every parameter can be used to control how often this method
        runs.

        Eg.
         - For DQN, synchronize target and value models
         - For REINFORCE do MC model training step
         - For random agent this is passed

         Note this there is no equivalent "during episode update" method as each agent defines it's own play_episode
         method, which can call model specific updates as needed (eg.f or DQN this sample from buffer + training
         step, for REINFORCE do nothing, etc.)
        """
        pass

    @staticmethod
    def _fake_tqdm(x: Any) -> Any:
        return x

    def _set_tqdm(self, verbose: bool = False) -> None:
        """Turn tqdm on or of depending on verbosity setting."""
        if verbose:
            self._tqdm = tqdm
        else:
            self._tqdm = self._fake_tqdm

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
        with open(fn, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fn: str) -> "AgentBase":
        with open(fn, 'rb') as f:
            new_agent = pickle.load(f)
        new_agent.check_ready()

        return new_agent
