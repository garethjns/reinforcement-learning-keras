"""Train and few LinearQAgents, plot the results, and run an episode on the best agent."""

from reinforcement_learning_keras.agents.q_learning.linear_q_agent import LinearQAgent
from reinforcement_learning_keras.enviroments.cart_pole.cart_pole_config import CartPoleConfig

from reinforcement_learning_keras.experiment.agent_experiment import AgentExperiment


def run_exp(n_episodes: int = 1000, max_episode_steps: int = 500):
    exp = AgentExperiment(agent_class=LinearQAgent,
                          agent_config=CartPoleConfig('linear_q'),
                          n_reps=32,
                          n_jobs=32,
                          training_options={"n_episodes": n_episodes,
                                            "max_episode_steps": max_episode_steps})

    exp.run()
    exp.save(fn=f"{LinearQAgent.__name__}_experiment.pkl")


if __name__ == "__main__":
    run_exp()
