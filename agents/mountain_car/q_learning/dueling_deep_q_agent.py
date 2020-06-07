from agents.agent_helpers.virtual_gpu import VirtualGPU
from agents.cart_pole.q_learning.dueling_deep_q_agent import DuelingDeepQAgent as CartDuelingDeepQAgent


class DuelingDeepQAgent(CartDuelingDeepQAgent):
    env_spec: str = "MountainCar-v0"
    learning_rate: float = 0.002

    @staticmethod
    def _final_reward(reward: float) -> float:
        return 500

    @classmethod
    def example(cls, n_episodes: int = 1000, render: bool = True) -> "DuelingDeepQAgent":
        """Run a quick example with n_episodes and otherwise default settings."""
        VirtualGPU(256)
        agent = cls("MountainCar-v0")
        agent.train(verbose=True, render=render,
                    max_episode_steps=1000,
                    n_episodes=n_episodes,
                    checkpoint_every=10)

        agent.save("dueling_dqn_mountain_car_example.pkl")

        return agent


if __name__ == "__main__":
    agent = DuelingDeepQAgent.example(render=True)
