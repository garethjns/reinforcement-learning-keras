"""Train and few LinearQAgents, plot the results, and run an episode on the best agent."""

from agents.mountain_car.q_learning.linear_q_agent import LinearQAgent

from experiment.agent_experiment import AgentExperiment


def run_exp(n_episodes: int = 500, max_episode_steps: int = 1000):
    exp = AgentExperiment(env_spec="MountainCar-v0",
                          agent_class=LinearQAgent,
                          n_reps=6,
                          n_jobs=6,
                          training_options={"n_episodes": n_episodes,
                                            "max_episode_steps": max_episode_steps})

    exp.run()
    exp.save(fn=f"{LinearQAgent.__name__}_experiment.pkl")


if __name__ == "__main__":
    run_exp()
