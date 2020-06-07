from agents.agent_helpers.virtual_gpu import VirtualGPU
from agents.cart_pole.q_learning.dueling_deep_q_agent import DuelingDeepQAgent as CartDuelingDeepQAgent


class DeepQAgent(CartDuelingDeepQAgent):
    env_spec: str = "MountainCar-v0"
    learning_rate: float = 0.001

    @staticmethod
    def _final_reward(reward: float) -> float:
        return 250

    @classmethod
    def example(cls, n_episodes: int = 5000, render: bool = True) -> "DeepQAgent":
        """Run a quick example with n_episodes and otherwise default settings."""
        VirtualGPU(128)
        agent = cls("MountainCar-v0")
        agent.train(verbose=True, render=render,
                    max_episode_steps=1000,
                    n_episodes=n_episodes,
                    checkpoint_every=False)

        return agent


if __name__ == "__main__":
    agent = DeepQAgent.example(render=True)
    agent.save("test_mountain_car_dqn.pkl")
