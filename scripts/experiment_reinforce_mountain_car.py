"""Train and few DeepQAgents, plot the results, and run an episode on the best agent."""
from reinforcement_learning_keras.agents.components.helpers.virtual_gpu import VirtualGPU
from reinforcement_learning_keras.agents.policy_gradient.reinforce_agent import ReinforceAgent
from reinforcement_learning_keras.enviroments.mountain_car.mountain_car_config import MountainCarConfig
from reinforcement_learning_keras.experiment.agent_experiment import AgentExperiment


def run_exp(n_episodes: int = 1000, max_episode_steps: int = 1000):
    gpu = VirtualGPU(256)

    exp = AgentExperiment(agent_class=ReinforceAgent,
                          agent_config=MountainCarConfig(agent_type='reinforce'),
                          n_reps=5,
                          n_jobs=1 if gpu.on else 5,
                          training_options={"n_episodes": n_episodes,
                                            "max_episode_steps": max_episode_steps,
                                            "update_every": 1})

    exp.run()
    exp.save(fn=f"{ReinforceAgent.__name__}_experiment.pkl")


if __name__ == "__main__":
    run_exp(n_episodes=1000)
