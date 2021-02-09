import abc
import copy
import gc
import os
import time
import warnings
from typing import Any, Callable, Union, Dict, Tuple, Iterable

import gym
import joblib
import numpy as np
from tf2_vgpu import VirtualGPU

from rlk.agents.components.helpers.env_builder import EnvBuilder
from rlk.agents.components.helpers.tqdm_handler import TQDMHandler
from rlk.agents.components.history.episode_report import EpisodeReport
from rlk.agents.components.history.training_history import TrainingHistory
from rlk.environments.config_base import ConfigBase


class AgentBase(abc.ABC):
    name: str
    env_spec: str
    env_kwargs: Dict[str, Any]
    env_builder: Union[EnvBuilder, None]
    env_wrappers: Iterable[Callable]
    env_kwargs: Union[None, Dict[str, Any]]
    env_builder_kwargs: Dict[str, Any]
    gamma: float
    final_reward: float
    ready: bool

    training_history: TrainingHistory
    _tqdm = TQDMHandler()

    @property
    def env(self) -> gym.Env:
        return self.env_builder.env

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

        # Get object spec to pickle, everything left except env_builder references
        # This is dodgy. Can end up with recursion errors depending on how deepcopy behaves...
        object_state_dict = copy.deepcopy({k: v for k, v in self.__dict__.items()})

        # Put this object back how it was
        self.check_ready()

        return object_state_dict

    def check_ready(self) -> None:
        """
        Check the model is ready to use.

        If super is used, should be at end of overloading method.

        Default implementation:
         - Check _env is set (most models?)
        Example of other model specific steps that might need doing:
         - For Keras models, check model is ready, for example if it needs recompiling after loading.
        """
        self.env_builder = EnvBuilder(env_spec=self.env_spec, env_wrappers=self.env_wrappers,
                                      env_kwargs=self.env_kwargs, **self.env_builder_kwargs)
        self.env_builder.set_env()
        self.ready = True
        gc.collect()

    def unready(self) -> None:
        """
        Remove anything that causes issues with pickling, such as keras models.

        If super is used, should be at end of overloading method.
        """
        if self.env_builder is not None:
            self.env_builder.env.close()
            self.env_builder = None
        self.ready = False
        gc.collect()

    def transform(self, s: Any) -> Any:
        """Run the any pre-preprocessing on raw state, if used."""
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
    def get_action(self, s: Any, **kwargs) -> Union[int, Tuple[int]]:
        """
        Given state s, get an action from the agent.

        May include other kwargs if needed - for example, a training flag for methods using epsilon greedy.

        May also return just action (eg in q learning) or Tuple[Action, (any, other, stuff)], for example for
        actor-critic where action, action_probs, and critic_value are returned when training.
        """
        pass

    @abc.abstractmethod
    def _play_episode(self, max_episode_steps: int = 500,
                      training: bool = False, render: bool = True) -> Tuple[float, int]:
        """
        Play a single episode with the agent (run multiple steps). Should return reward and n frames.

        :param max_episode_steps: Max steps before stopping, overrides any time limit set by Gym.
        :param training: Bool to indicate whether or not to use this experience to update the model.
        :param render: Bool to indicate whether or not to call env.render() each training step.
        :return: The total real reward for the episode and number of frames run.
        """
        pass

    def play_episode(self, max_episode_steps: int = 500,
                     training: bool = False, render: bool = True) -> EpisodeReport:
        """
        Run Agent's _play_episode and produce episode report.

        :param max_episode_steps: Max steps before stopping, overrides any time limit set by Gym.
        :param training: Bool to indicate whether or not to use this experience to update the model.
        :param render: Bool to indicate whether or not to call env.render() each training step.
        :return: The total real reward for the episode, number of frames run, and time taken in seconds.
        """
        t0 = time.time()
        total_reward, frames = self._play_episode(max_episode_steps=max_episode_steps, training=training, render=render)
        t1 = time.time()

        return EpisodeReport(total_reward=total_reward,
                             frames=frames,
                             time_taken=np.round(t1 - t0, 3),
                             epsilon_used=getattr(self, 'eps', None))

    def train(self, n_episodes: int = 10000, max_episode_steps: int = 500, verbose: bool = True, render: bool = True,
              checkpoint_every: Union[bool, int] = 0, update_every: Union[bool, int] = 1) -> None:
        """
        Run the default training loop

        :param n_episodes: Number of episodes to run.
        :param max_episode_steps: Max steps before stopping, overrides any time limit set by Gym.
        :param verbose:  If verbose, use tqdm and print last episode score for feedback during training.
        :param render: Bool to indicate whether or not to call env.render() each training step.
        :param checkpoint_every: Save the model every n steps while training. Set to 0 or false to turn off.
        :param update_every: Run the _after_episode_update() step every n episodes.
        """
        self._tqdm.set_tqdm(verbose)

        for ep in self._tqdm.tqdm_runner(range(n_episodes)):
            episode_report = self.play_episode(max_episode_steps=max_episode_steps, training=True, render=render)
            self._update_history(episode_report, verbose)

            if (update_every > 0) and not (ep % update_every):
                # Run the after-episode update step
                self._after_episode_update()

            if (checkpoint_every > 0) and (ep > 0) and (not ep % checkpoint_every):
                self.save()

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

    def _update_history(self, episode_report: EpisodeReport, verbose: bool = True) -> None:
        """
        Add an episodes reward to history and maybe plot depending on history settings.

        :param episode_report: Episode report to add to history.
        :param verbose: If verbose, print the last episode and run the history plot. The history plot will display
                        depending on it's own settings. Verbose = False will turn it off totally.
        """
        self.training_history.append(episode_report)

        if verbose:
            print(f"{self.name}: {episode_report}")
            self.training_history.training_plot()

    def get_weights(self) -> np.ndarray:
        """Get model weights."""
        raise NotImplementedError

    def set_weights(self, weights: np.ndarray) -> None:
        """Set model weights."""
        raise NotImplementedError

    def save(self) -> None:
        with open(f"{self.name}_{self.env_spec}", 'wb') as f:
            joblib.dump(self, f)

    @classmethod
    def load(cls, fn: str) -> "AgentBase":
        with open(fn, 'rb') as f:
            new_agent = joblib.load(f)
        new_agent.check_ready()

        return new_agent

    @classmethod
    def example(cls, config: ConfigBase, render: bool = True,
                n_episodes: int = 500, max_episode_steps: int = 500,
                checkpoint_every: int = 100, **kwargs) -> "AgentBase":
        """
        Default example runner.

        kwargs should contain algo specific settings and will be passed to .train. eg, update_every for dqn.
        """
        VirtualGPU(config.gpu_memory)

        config_dict = config.build()
        if os.path.exists(config_dict['name']):
            agent = cls.load(config_dict['name'])
            warnings.warn('Loaded existing agent.')
        else:
            agent = cls(**config_dict)  # noqa - __init__ will be available in non abstract children

        agent.train(verbose=True, render=render,
                    n_episodes=n_episodes, max_episode_steps=max_episode_steps,
                    checkpoint_every=checkpoint_every, **kwargs)
        agent.save()

        return agent
