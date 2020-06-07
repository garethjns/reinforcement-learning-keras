import copy
import pickle
import warnings
from dataclasses import dataclass
from typing import Callable, Union, Dict, Any
from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import BrokenProcessPool

from agents.agent_base import AgentBase


@dataclass
class AgentExperiment:
    env_spec: str
    agent_class: Callable
    name: str = "unnamed_experiment"
    n_reps: int = 5
    n_jobs: int = 1
    training_options: Union[None, Dict[str, Any]] = None

    def __post_init__(self):
        self._trained_agents: List[AgentBase] = []
        self._set_default_training_options()

    def _set_default_training_options(self):
        if self.training_options is None:
            self.training_options = {}

        defaults = {"n_episodes": 500, "max_episode_steps": 500, "render": False, "verbose": False}
        for k, v in defaults.items():
            if k not in self.training_options:
                self.training_options[k] = defaults[k]

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
    def _fit_agent(agent_class: Callable, env_spec: str, training_options: Dict[str, Any]):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)

            agent = agent_class(env_spec=env_spec)
            agent.train(**training_options)
            agent.unready()

        return agent

    def _run(self) -> None:
        self._trained_agents = Parallel(
            backend='loky', verbose=10,
            n_jobs=self.n_jobs)(delayed(self._fit_agent)(self.agent_class, self.env_spec, self.training_options)
                                for _ in range(self.n_reps))

    def run(self) -> None:
        try:
            # All agents support pickling, so joblib can run some in parallel...
            self._run()
        except BrokenProcessPool:
            # ... Except for TF models running on GPU, they'll probably crap out. Run 1 by 1.
            # OR it'll crash Chrome, and Python, and hang for all eternity. It's best not to rely on this.
            self.n_jobs = 1
            self._run()

        self.plot()
        self.play_best()

    def plot(self) -> None:
        sns.set()

        full_history = np.hstack([np.vstack(a.history.history) for a in self._trained_agents])
        y_mean = np.mean(full_history, axis=1)
        y_std = np.std(full_history, axis=1)

        plt.plot(y_mean, label='Mean score', lw=1.25)
        # 5% moving avg
        mv_avg_pts = max(1, int(len(y_mean) * 0.05))
        plt.plot(np.convolve(self.best_agent.history.history, np.ones(mv_avg_pts), 'valid') / mv_avg_pts,
                 label='Best (mv avg)', ls='--', color='#d62728', lw=0.5)
        plt.plot(np.convolve(self.worst_agent.history.history, np.ones(mv_avg_pts), 'valid') / mv_avg_pts,
                 label='Worst (mv avg)', ls='--', color='#9467bd', lw=0.5)
        plt.fill_between(range(len(y_mean)),
                         [max(0, s) for s in y_mean - y_std],
                         [min(len(y_mean), s) for s in y_mean + y_std],
                         color='lightgray', label='Score std')

        plt.title(f'{self._trained_agents[0].name}', fontweight='bold')
        plt.xlabel('N episodes', fontweight='bold')
        plt.ylabel('Score', fontweight='bold')
        plt.legend(title='Agents')
        plt.tight_layout()
        plt.savefig(f'{self.name}_{self.env_spec}_{self._trained_agents[0].name}.png')

    def play_best(self, episode_steps: int = 500):
        best_agent = copy.deepcopy(self.best_agent)
        best_agent.check_ready()
        best_agent._env_builder.set_env(gym.wrappers.Monitor(best_agent.env,
                                                             f'{self._trained_agents[0].name}_monitor_dir',
                                                             force=True))

        try:
            best_agent.play_episode(training=False, render=False, max_episode_steps=episode_steps)
        except (ImportError, gym.error.DependencyNotInstalled) as e:
            print(f"Monitor wrapper failed, not saving video: \n{e}")

    def save(self, fn: str):
        pickle.dump(self, open(fn, 'wb'))

    def save_best_agent(self, fn: str = None):
        if fn is None:
            fn = f"{self.best_agent.name}.pkl"

        pickle.dump(self.best_agent, open(fn, 'wb'))
