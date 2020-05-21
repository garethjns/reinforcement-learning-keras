"""Train and few DeepQAgents, plot the results, and run an episode on the best agent."""

from agents.cart_pole.q_learning.deep_q_agent import DeepQAgent

from experiment.agent_experiment import AgentExperiment


def run_exp(n_episodes: int = 500, max_episode_steps: int = 500):
    gpu = DeepQAgent.set_tf(512)

    exp = AgentExperiment(env_spec="CartPole-v0",
                          agent_class=DeepQAgent,
                          n_reps=5,
                          n_jobs=1 if gpu else -2,
                          n_episodes=n_episodes,
                          max_episode_steps=max_episode_steps)

    exp.run()


if __name__ == "__main__":
    run_exp(n_episodes=500)
