"""
This script evaluates the rl agent (and also pretrained agent for comparison) against the built in AI in the GFootball
environment.
"""

from collections import Callable
from typing import List

import numpy as np
from joblib import Parallel, delayed
from kaggle_football.envs.register_environments import register_all, a_11_vs_11_easy_stochastic

from scripts.gfootball.kaggle_agent_classification_model import agent as classification_model
from scripts.gfootball.kaggle_agent_rl_model import agent as rl_model

register_all()

N_GAMES = 2
N_JOBS = 2


def play(agent: Callable, render: bool = False) -> List:
    # Play agent vs ai
    env = a_11_vs_11_easy_stochastic()

    if render:
        env.render()
    obs = env.reset()

    rewards = []
    for _ in range(3000):
        obs, reward, _, _ = env.step(agent({'players_raw': obs}))
        rewards.append(reward)

    return rewards


def play_multiple(agent: Callable):
    games = Parallel(n_jobs=N_JOBS)(delayed(play)(agent) for _ in range(N_GAMES))

    games_rewards = [sum(rewards) for rewards in games]

    return np.mean(games_rewards), np.std(games_rewards)


if __name__ == "__main__":

    # TODO: Add open_rules_bot
    for name, agent in zip(['classification_model', 'rl_model'], [classification_model, rl_model]):
        mean, std = play_multiple(agent)
        print(f"\n {name} vs AI:\n"
              f"Left agent mean +/- std goal diff {mean} +/- {std}")

        play_multiple(classification_model)
        play_multiple(rl_model)
