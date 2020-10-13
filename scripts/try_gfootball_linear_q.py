import warnings

from sklearn.exceptions import DataConversionWarning

from reinforcement_learning_keras.agents.q_learning.linear_q_agent import LinearQAgent
from reinforcement_learning_keras.environments.gfootball.gfootball_config import GFootballConfig

if __name__ == "__main__":
    agent = LinearQAgent(**GFootballConfig('linear_q').build())

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DataConversionWarning)
        agent.train(verbose=True, render=False,
                    n_episodes=200, max_episode_steps=1000, update_every=5,
                    checkpoint_every=0)
