from reinforcement_learning_keras.agents.q_learning.deep_q_agent import DeepQAgent
from reinforcement_learning_keras.environments.gfootball.gfootball_config import GFootballConfig

if __name__ == "__main__":
    agent = DeepQAgent(**GFootballConfig('dqn', env_spec="GFootball-11_vs_11_kaggle-SMM-v0").build())

    agent.train(verbose=True, render=False,
                n_episodes=1000, max_episode_steps=10000, update_every=10,
                checkpoint_every=10)
