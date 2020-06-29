"""Train and few DeepQAgents, plot the results, and run an episode on the best agent."""
from agents.components.helpers.virtual_gpu import VirtualGPU
from agents.q_learning.deep_q_agent import DeepQAgent
from enviroments.atari.pong.pong_config import PongConfig
from experiment.agent_experiment import AgentExperiment


def run_exp(agent_type: str, n_episodes: int = 400, max_episode_steps: int = 10000):
    gpu = VirtualGPU(2048)

    exp = AgentExperiment(name=f"{agent_type} Pong",
                          agent_class=DeepQAgent,
                          agent_config=PongConfig(agent_type=agent_type),
                          n_reps=5,
                          n_jobs=1 if gpu.on else 5,
                          training_options={"n_episodes": n_episodes,
                                            "verbose": 1,
                                            "max_episode_steps": max_episode_steps})

    exp.run()
    exp.save(fn=f"{DeepQAgent.__name__}_{agent_type}experiment.pkl")


if __name__ == "__main__":
    run_exp(agent_type='dqn')
    run_exp(agent_type='dueling_dqn')
    run_exp(agent_type='double_dqn')
    run_exp(agent_type='double_dueling_dqn')
