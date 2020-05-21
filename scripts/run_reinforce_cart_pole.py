"""Train and few DeepQAgents, plot the results, and run an episode on the best agent."""

from agents.cart_pole.policy_gradient.reinforce_agent import ReinforceAgent

from experiment.agent_experiment import AgentExperiment


def run_exp(n_episodes: int = 500, max_episode_steps: int = 500):
    gpu = ReinforceAgent.set_tf(256)

    exp = AgentExperiment(env_spec="CartPole-v0",
                          agent_class=ReinforceAgent,
                          n_reps=5,
                          n_jobs=1 if gpu else -1,
                          n_episodes=n_episodes,
                          max_episode_steps=max_episode_steps)

    exp.run()


if __name__ == "__main__":
    run_exp(n_episodes=500)
