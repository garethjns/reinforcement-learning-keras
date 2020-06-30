"""Train and few LinearQAgents, plot the results, and run an episode on the best agent."""
from reinforcement_learning_keras.agents.q_learning.linear_q_agent import LinearQAgent
from reinforcement_learning_keras.enviroments.mountain_car.mountain_car_config import MountainCarConfig
from reinforcement_learning_keras.experiment.agent_experiment import AgentExperiment


def run_exp(n_episodes: int = 500, max_episode_steps: int = 1000):
    exp = AgentExperiment(agent_class=LinearQAgent,
                          agent_config=MountainCarConfig(agent_type='linear_1'),
                          n_reps=6,
                          n_jobs=6,
                          training_options={"n_episodes": n_episodes,
                                            "max_episode_steps": max_episode_steps})

    exp.run()
    exp.save(fn=f"{LinearQAgent.__name__}_experiment.pkl")


if __name__ == "__main__":
    run_exp(n_episodes=500)
