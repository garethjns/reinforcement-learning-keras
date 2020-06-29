"""Train and few DeepQAgents, plot the results, and run an episode on the best agent."""
from agents.components.helpers.virtual_gpu import VirtualGPU
from agents.q_learning.deep_q_agent import DeepQAgent
from enviroments.mountain_car.mountain_car_config import MountainCarConfig
from experiment.agent_experiment import AgentExperiment


def run_exp(agent_type: str, n_episodes: int = 500, max_episode_steps: int = 1000):
    gpu = VirtualGPU(256)

    exp = AgentExperiment(name=f"{agent_type} MountainCar",
                          agent_class=DeepQAgent,
                          n_reps=8,
                          n_jobs=1 if gpu.on else 8,
                          training_options={"n_episodes": n_episodes,
                                            "max_episode_steps": max_episode_steps})

    exp.run()
    exp.save(fn=f"{DeepQAgent.__name__}_{agent_type}experiment.pkl")


if __name__ == "__main__":
    run_exp(agent_type='dqn')
    run_exp(agent_type='dueling_dqn')
    run_exp(agent_type='double_dqn')
    run_exp(agent_type='double_dueling_dqn')
