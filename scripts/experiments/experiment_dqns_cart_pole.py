"""Train and few DeepQAgents, plot the results, and run an episode on the best agent."""
from rlk.agents.q_learning.deep_q_agent import DeepQAgent
from rlk.environments.cart_pole.cart_pole_config import CartPoleConfig
from rlk.experiment.agent_experiment import AgentExperiment


def run_exp(agent_type: str, n_episodes: int = 1000, max_episode_steps: int = 500):
    exp = AgentExperiment(name=f"{agent_type} CartPole",
                          agent_class=DeepQAgent,
                          agent_config=CartPoleConfig(agent_type=agent_type),
                          n_reps=6,
                          n_jobs=6,
                          training_options={"n_episodes": n_episodes,
                                            "max_episode_steps": max_episode_steps})

    exp.run()
    exp.save(fn=f"{DeepQAgent.__name__}_{agent_type}experiment.pkl")


if __name__ == "__main__":
    run_exp(agent_type='dqn')
    run_exp(agent_type='dueling_dqn')
    run_exp(agent_type='double_dqn')
    run_exp(agent_type='double_dueling_dqn')
