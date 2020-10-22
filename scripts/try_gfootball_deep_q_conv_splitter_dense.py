from rlk.agents.q_learning.deep_q_agent import DeepQAgent
from rlk.environments.gfootball.gfootball_config import GFootballConfig

if __name__ == "__main__":
    agent = DeepQAgent(**GFootballConfig('dqn', env_spec="GFootball-11_vs_11_kaggle-SMM-v0").build())

    agent.train(verbose=True, render=True, checkpoint_every=100,
                n_episodes=20000, max_episode_steps=3000, update_every=1)
