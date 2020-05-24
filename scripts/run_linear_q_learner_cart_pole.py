"""Train and few LinearQAgents, plot the results, and run an episode on the best agent."""

from agents.cart_pole.q_learning.linear_q_agent import LinearQAgent

from experiment.agent_experiment import AgentExperiment


def run_exp(n_episodes: int = 1000, max_episode_steps: int = 500):
    exp = AgentExperiment(env_spec="CartPole-v0",
                          agent_class=LinearQAgent,
                          n_reps=8,
                          n_jobs=4,
                          n_episodes=n_episodes,
                          max_episode_steps=max_episode_steps)

    exp.run()
    exp.save(fn=f"{LinearQAgent.__name__}_experiment.pkl")

if __name__ == "__main__":
    run_exp()
