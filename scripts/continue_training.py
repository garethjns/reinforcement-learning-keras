import os

from rlk.agents.components.helpers.virtual_gpu import VirtualGPU
from rlk.agents.q_learning.deep_q_agent import DeepQAgent
import vizdoomgym  # noqa

VirtualGPU(512)

agent = DeepQAgent.load("DoubleDuelingDQN_GFootball-academy_empty_goal_close-v0")

agent.train(n_episodes=10000, max_episode_steps=20000, checkpoint_every=100, render=False)
