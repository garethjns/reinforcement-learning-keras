import copy
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

from reinforcement_learning_keras.agents.agent_base import AgentBase
from reinforcement_learning_keras.enviroments.config_base import ConfigBase


@dataclass
class AgentExperiment:
    agent_class: Callable
    agent_config: ConfigBase
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
    def agent_scores(self) -> List[float]:
        return [a.training_history.current_performance for a in self._trained_agents]

    @property
    def best_agent(self) -> AgentBase:
        return self._trained_agents[int(np.argmax(self.agent_scores))]

    @property
    def worst_agent(self) -> AgentBase:
        return self._trained_agents[int(np.argmin(self.agent_scores))]

    @staticmethod
    def _fit_agent(agent_class: Callable, agent_config: ConfigBase, training_options: Dict[str, Any]):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)

            config_dict = agent_config.build()
            # Give each agent a unique name for easier tracking with verbose and multiprocessing
            config_dict["name"] = f"{config_dict.get('name', 'Agent')}_{np.random.randint(0, 2 ** 16)}"

            agent = agent_class(**config_dict)
            agent.train(**training_options)
            # Might as well save agent. This will also unready and save buffers, models, etc.
            agent.save()
            agent.unready()

        return agent

    def _run(self) -> None:
        self._trained_agents = Parallel(
            backend='loky', verbose=10,
            n_jobs=self.n_jobs)(delayed(self._fit_agent)(self.agent_class, self.agent_config, self.training_options)
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
        self.save_best_agent()

    def plot(self, err: str = 'range') -> None:
        """
        Plot reward vs episode for experiment.

        Plots:
        - Mean of all agents
        - Std or range of all agents (not constrained by max episodes or minimum score)
        - Min and max score across all agents for each episode (observed; constrained by max episodes or minimum score)
        - Score of best and worst agents (with 5% moving average). Best and worst defined using "current_performance"
          property of agents, which is mean score over most recent n episodes, where n is whatever the rolling average
          specified in the agents training history was.
        """

        sns.set()

        full_history = np.hstack([np.vstack(a.training_history.get_metric('total_reward'))
                                  for a in self._trained_agents])

        # Summary stats
        n_episodes = full_history.shape[0]
        y_mean = np.mean(full_history, axis=1)

        plt.plot(y_mean, color='#1f77b4', label='Mean score', lw=1.25)
        if err == 'range':
            plt.fill_between(range(n_episodes), np.min(full_history, axis=1), np.max(full_history, axis=1),
                             color='lightgrey', label='Score range', alpha=0.5)
        else:
            y_std = np.std(full_history, axis=1)
            plt.fill_between(range(n_episodes), y_mean - y_std, y_mean + y_std,
                             color='#1f77b4', label='Score std', alpha=0.3)

        # Best and worst agents
        mv_avg_pts = max(1, int(n_episodes * 0.05))  # 5% moving avg
        plt.plot(np.convolve(self.best_agent.training_history.get_metric('total_reward'),
                             np.ones(mv_avg_pts), 'valid') / mv_avg_pts,
                 label='Best (mv avg)', ls='--', color='#d62728', lw=0.7)
        plt.plot(np.convolve(self.worst_agent.training_history.get_metric('total_reward'),
                             np.ones(mv_avg_pts), 'valid') / mv_avg_pts,
                 label='Worst (mv avg)', ls='--', color='#9467bd', lw=0.7)

        plt.title(f'{self.name}', fontweight='bold')
        plt.xlabel('Episode', fontweight='bold')
        plt.ylabel('Score', fontweight='bold')
        plt.legend(title='Agents')
        plt.tight_layout()
        plt.savefig(f'{self.name}_{self.agent_config.env_spec}.png')
        plt.close()

    def play_best(self, episode_steps: int = None):
        if episode_steps is None:
            episode_steps = self.training_options["max_episode_steps"]

        best_agent = copy.deepcopy(self.best_agent)
        best_agent.check_ready()
        best_agent.env_builder.set_env(gym.wrappers.Monitor(best_agent.env,
                                                            f'{self._trained_agents[0].name}_monitor_dir',
                                                            force=True))

        try:
            best_agent.play_episode(training=False, render=False, max_episode_steps=episode_steps)
        except (ImportError, gym.error.DependencyNotInstalled) as e:
            print(f"Monitor wrapper failed, not saving video: \n{e}")

    def save(self, fn: str):
        """Disabled for now... Needed?"""
        pass

    def save_best_agent(self):
        self.best_agent.save()
