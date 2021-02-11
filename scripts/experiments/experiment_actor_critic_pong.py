"""Train and few DeepQAgents, plot the results, and run an episode on the best agent."""
from rlk.agents.actor_critic.actor_critic import ActorCriticAgent
from rlk.agents.q_learning.deep_q_agent import DeepQAgent
from rlk.environments.atari.pong.pong_config import PongConfig
from rlk.experiment.agent_experiment import AgentExperiment


def run_exp(agent_type: str, n_episodes: int = 400, max_episode_steps: int = 10000,
            model_mode: str = 'diff'):
    config = PongConfig(agent_type=agent_type, mode=model_mode)

    exp = AgentExperiment(name=f"{agent_type} {model_mode} Pong",
                          agent_class=ActorCriticAgent,
                          agent_config=config,
                          n_reps=3,
                          n_jobs=3,
                          gpu_memory_per_agent=1024,
                          training_options={"n_episodes": n_episodes,
                                            "verbose": 1,
                                            "max_episode_steps": max_episode_steps})

    exp.run()
    exp.save(fn=f"{DeepQAgent.__name__}_{agent_type}experiment.pkl")


if __name__ == "__main__":
    run_exp(agent_type='actor_critic')
