"""Train and few DeepQAgents, plot the results, and run an episode on the best agent."""
from agents.components.helpers.virtual_gpu import VirtualGPU
from agents.q_learning.deep_q_agent import DeepQAgent
from enviroments.pong.pong_config import PongConfig
from experiment.agent_experiment import AgentExperiment


def run_exp(n_episodes: int = 500, max_episode_steps: int = 10000):
    gpu = VirtualGPU(4096)

    exp = AgentExperiment(agent_class=DeepQAgent,
                          agent_config=PongConfig(agent_type='dqn'),
                          n_reps=3,
                          n_jobs=1 if gpu.on else 3,
                          training_options={"n_episodes": n_episodes,
                                            'verbose': True,
                                            "max_episode_steps": max_episode_steps})

    exp.run()
    exp.save(fn=f"{DeepQAgent.__name__}_experiment.pkl")


if __name__ == "__main__":
    run_exp(n_episodes=500)