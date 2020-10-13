import gfootball  # noqa
import gym

from reinforcement_learning_keras.agents.q_learning.deep_q_agent import DeepQAgent
from reinforcement_learning_keras.environments.gfootball.gfootball_config import GFootballConfig

if __name__ == "__main__":
    env_name = "GFootball-SimpleSMM-v0"
    gym.envs.register(
        id=env_name, max_episode_steps=10000,
        entry_point="reinforcement_learning_keras.environments.gfootball.kaggle_environment:kaggle_environment")

    gym.make(env_name)

    agent = DeepQAgent(**GFootballConfig('double_dqn', env_spec=env_name).build())

    agent.train(verbose=True, render=False,
                n_episodes=20000, max_episode_steps=3000, update_every=1,
                checkpoint_every=100)
