"""Train and few LinearQAgents, plot the results, and run an episode on the best agent."""

import tensorflow as tf
from joblib.externals.loky.process_executor import TerminatedWorkerError

from agents.cart_pole.q_learning.deep_q_learning_agent import DQNAgent
from scripts.run_linear_q_learner_cart_pole import train_all, plot_all, play_best

if __name__ == "__main__":
    try:
        # Handle running on GPU: If available, reduce memory commitment to avoid over-committing error in 2.2.0 and
        # for also for convenience.
        tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=1024)])
    except (IndexError, AttributeError):
        pass

    common_kwargs = {"agent_class": DQNAgent,
                     "n_agents": 10,
                     "n_episodes": 1000,
                     "max_episode_steps": 500}
    try:
        # Try and run with n_jobs > 1. Might work, probably won't. If running on GPU it'll probably break. If running
        # on CPU it'll work if the compiled Keras models can be pickled (they probably can't).
        agents_ = train_all(n_jobs=4, **common_kwargs)
    except (TerminatedWorkerError, TypeError):
        # Give up and run in slow mode.
        agents_ = train_all(n_jobs=1, **common_kwargs)

    plot_all(agents_)
    play_best(agents_)
