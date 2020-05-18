"""Train and few LinearQAgents, plot the results, and run an episode on the best agent."""

import tensorflow as tf

from agents.cart_pole.q_learning.deep_q_learning_agent import DQNAgent
from scripts.run_linear_q_learner_cart_pole import train_all, plot_all, play_best

if __name__ == "__main__":
    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=1024)])

    agents_ = train_all(agent_class=DQNAgent,
                        n_agents=4,
                        n_jobs=1,
                        n_episodes=10,
                        max_episode_steps=500)

    plot_all(agents_)
    play_best(agents_)
