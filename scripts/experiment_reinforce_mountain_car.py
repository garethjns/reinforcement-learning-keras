"""Train and few DeepQAgents, plot the results, and run an episode on the best agent."""

from agents.mountain_car.policy_gradient.reinforce_agent import ReinforceAgent
from agents.agent_helpersvirtual_gpu import VirtualGPU

from experiment.agent_experiment import AgentExperiment


def run_exp(n_episodes: int = 1000, max_episode_steps: int = 1000):
    gpu = VirtualGPU(256)

    exp = AgentExperiment(env_spec="MountainCar-v0",
                          agent_class=ReinforceAgent,
                          n_reps=5,
                          n_jobs=1 if gpu.on else 5,
                          training_options={"n_episodes": n_episodes,
                                            "max_episode_steps": max_episode_steps,
                                            "update_every": 1})

    exp.run()
    exp.save(fn=f"{ReinforceAgent.__name__}_experiment.pkl")


if __name__ == "__main__":
    run_exp(n_episodes=1000)
