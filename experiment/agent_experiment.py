import copy
import warnings
from dataclasses import dataclass
from typing import Callable
from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed

from agents.agent_base import AgentBase


@dataclass
class AgentExperiment:
    env_spec: str
    agent_class: Callable
    n_reps: int = 5
    n_jobs: int = 1
    n_episodes: int = 500
    max_episode_steps: int = 500

    def __post_init__(self):
        self._trained_agents: List[AgentBase] = []

    @property
    def agent_scores(self):
        return [a.history.current_performance for a in self._trained_agents]

    @property
    def best_agent(self) -> AgentBase:
        return self._trained_agents[int(np.argmax(self.agent_scores))]

    @property
    def worst_agent(self) -> AgentBase:
        return self._trained_agents[int(np.argmin(self.agent_scores))]

    @staticmethod
    def _fit_agent(agent_class: Callable,
                   n_episodes: int = 500, max_episode_steps: int = 500):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)

            agent = agent_class(env_spec="CartPole-v0")
            agent.train(n_episodes=n_episodes, max_episode_steps=max_episode_steps, verbose=False, render=False)
            agent.unready()

        return agent

    def _run(self) -> None:
        self._trained_agents = Parallel(
            backend='loky',
            verbose=10,
            n_jobs=self.n_jobs)(delayed(self._fit_agent)(self.agent_class, self.n_episodes, self.max_episode_steps)
                                for _ in range(self.n_reps))

    def run(self) -> None:
        self._run()
        self.plot()
        self.play_best()

    def plot(self) -> None:
        sns.set()

        full_history = np.hstack([np.vstack(a.history.history) for a in self._trained_agents])
        y_mean = np.mean(full_history, axis=1)
        y_std = np.std(full_history, axis=1)

        plt.plot(self.best_agent.history.history, label='Best', ls='--', color='#d62728', lw=0.35)
        plt.plot(self.worst_agent.history.history, label='Worst', ls='--', color='#9467bd', lw=0.35)
        plt.plot(y_mean, label='Mean score', lw=2.5)
        plt.fill_between(range(len(y_mean)),
                         [max(0, s) for s in y_mean - y_std],
                         [min(500, s) for s in y_mean + y_std],
                         color='lightgray', label='Score std')
        plt.title(f'{self._trained_agents[0].name}', fontweight='bold')
        plt.xlabel('N episodes', fontweight='bold')
        plt.ylabel('Score', fontweight='bold')
        plt.legend(title='Agents')
        plt.tight_layout()
        plt.savefig(f'{self._trained_agents[0].name}.png')

    def play_best(self):
        best_agent = copy.deepcopy(self.best_agent)
        best_agent.check_ready()
        best_agent._env = gym.wrappers.Monitor(best_agent._env, 'monitor_dir', force=True)
        try:
            best_agent.play_episode(training=False, render=False, max_episode_steps=self.max_episode_steps)
        except gym.error.DependencyNotInstalled as e:
            print(f"Monitor wrapper failed, not saving video: \n{e}")
