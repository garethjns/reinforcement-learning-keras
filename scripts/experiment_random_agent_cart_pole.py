"""Train and few LinearQAgents, plot the results, and run an episode on the best agent."""

from reinforcement_learning_keras.agents.random.random_agent import RandomAgent
from reinforcement_learning_keras.enviroments.cart_pole.cart_pole_config import CartPoleConfig

from reinforcement_learning_keras.experiment.agent_experiment import AgentExperiment


def run_exp(n_episodes: int = 1000, max_episode_steps: int = 500):
    exp = AgentExperiment(agent_class=RandomAgent,
                          agent_config=CartPoleConfig('random'),
                          n_reps=8,
                          n_jobs=4,
                          training_options={"n_episodes": n_episodes,
                                            "max_episode_steps": max_episode_steps})

    exp.run()
    exp.save(fn=f"{RandomAgent.__name__}_experiment.pkl")


if __name__ == "__main__":
    run_exp()
