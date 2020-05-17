"""Train and few LinearQAgents, plot the results, and run an episode on the best agent."""
import warnings
from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed

from agents.cart_pole.q_learning.components.epsilon_greedy import EpsilonGreedy
from agents.cart_pole.q_learning.linear_q_learning_agent import LinearQLearningAgent


def fit_agent():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)

        env = gym.make("CartPole-v0")
        eps = EpsilonGreedy(eps_initial=0.5, eps_min=0.01)
        agent = LinearQLearningAgent(env, eps=eps)
        agent.train(n_episodes=1000, max_episode_steps=500, verbose=False, render=False)

    return agent


def train_all(n_agents: int = 5, n_jobs: int = -2) -> List[LinearQLearningAgent]:
    agents = Parallel(backend='loky',
                      verbose=10,
                      n_jobs=n_jobs)(delayed(fit_agent)()
                                     for _ in range(n_agents))

    return agents


def plot_all(agents: List[LinearQLearningAgent]) -> None:
    hist = np.hstack([np.vstack(a.history.history) for a in agents])

    sns.set()
    y_mean = np.mean(hist, axis=1)
    y_std = np.std(hist, axis=1)
    plt.plot(y_mean)
    plt.fill_between(range(len(y_mean)), y_mean - y_std, y_mean + y_std,
                     color='lightgray')
    plt.title('LinearQAgent', fontweight='bold')
    plt.xlabel('N episodes', fontweight='bold')
    plt.ylabel('Score', fontweight='bold')
    plt.tight_layout()
    plt.savefig('LinearQAgent.png')


def play_best(agents: List[LinearQLearningAgent]):
    scores = [a.history.current_performance for a in agents]
    best_agent = agents[int(np.argmax(scores))]

    best_agent.env = gym.wrappers.Monitor(best_agent.env, 'monitor_dir', force=True)
    best_agent.play_episode(training=False, render=False, max_episode_steps=100)


if __name__ == "__main__":
    agents_ = train_all(n_agents=8,
                        n_jobs=4)

    plot_all(agents_)
    play_best(agents_)
