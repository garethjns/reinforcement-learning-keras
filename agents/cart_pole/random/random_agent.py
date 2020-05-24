from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.agent_base import AgentBase
from agents.plotting.training_history import TrainingHistory


@dataclass
class RandomAgent(AgentBase):
    env_spec: str = "CartPole-v0"
    name: str = 'RandomAgent'
    plot_during_training: bool = True

    def __post_init__(self):
        self.history = TrainingHistory(plotting_on=self.plot_during_training,
                                       plot_every=25,
                                       rolling_average=12,
                                       agent_name=self.name)
        self._set_env()
        self._build_model()

    def _model_f(self):
        return int(np.random.randint(0, self._env.action_space.n, 1))

    def _build_model(self) -> None:
        """Set model function. Note using a lambda breaks pickle support."""
        self.model = self._model_f

    def update_model(self, *args, **kwargs) -> None:
        pass

    def get_action(self, s: Any, **kwargs) -> int:
        return self.model()

    def play_episode(self, max_episode_steps: int = 500,
                     training: bool = False, render: bool = True) -> float:
        """
        Play a single episode and return the total reward.

        :param max_episode_steps: Max steps before stopping, overrides any time limit set by Gym.
        :param training: Bool to indicate whether or not to use this experience to update the model.
        :param render: Bool to indicate whether or not to call env.render() each training step.
        :return: The total real reward for the episode.
        """
        self._env._max_episode_steps = max_episode_steps
        _ = self._env.reset()
        total_reward = 0
        for _ in range(max_episode_steps):
            action = self.get_action(None)
            _, reward, done, _ = self._env.step(action)
            total_reward += reward

            if render:
                self._env.render()

            if done:
                break

        return total_reward

    def train(self, n_episodes: int = 10000, max_episode_steps: int = 500,
              verbose: bool = True, render: bool = True) -> None:
        """
        Run the training loop

        :param n_episodes: Number of episodes to run.
        :param max_episode_steps: Max steps before stopping, overrides any time limit set by Gym.
        :param verbose:  If verbose, use tqdm and print last episode score for feedback during training.
        :param render: Bool to indicate whether or not to call env.render() each training step.
        """
        self._set_tqdm(verbose)

        for _ in self._tqdm(range(n_episodes)):
            total_reward = self.play_episode(max_episode_steps,
                                             training=True, render=render)
            self._update_history(total_reward, verbose)

    @classmethod
    def example(cls, n_episodes: int = 1000, render: bool = True) -> "RandomAgent":
        agent = cls("CartPole-v0")
        agent.train(verbose=True, render=render,
                    n_episodes=n_episodes)

        return agent
