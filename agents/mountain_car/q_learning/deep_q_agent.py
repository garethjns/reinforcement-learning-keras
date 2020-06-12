from dataclasses import dataclass

from agents.agent_helpers.virtual_gpu import VirtualGPU
from agents.cart_pole.q_learning.deep_q_agent import DeepQAgent as CartDeepQAgent


@dataclass
class DeepQAgent(CartDeepQAgent):
    env_spec: str = "MountainCar-v0"
    learning_rate: float = 0.0025

    @staticmethod
    def _final_reward(reward: float) -> float:
        return 650

    @classmethod
    def example(cls, n_episodes: int = 500, render: bool = True) -> "DeepQAgent":
        """Run a quick example with n_episodes and otherwise default settings."""
        VirtualGPU(128)
        agent = cls()
        agent.train(verbose=True, render=render,
                    max_episode_steps=1000,
                    n_episodes=n_episodes,
                    checkpoint_every=20)

        agent.save("dqn_mountain_car_example.pkl")

        return agent


if __name__ == "__main__":
    agent_ = DeepQAgent.example(render=False)
    agent_.save("deep_q_agent_mountain_car.pkl")
