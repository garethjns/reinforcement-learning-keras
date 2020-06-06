from agents.cart_pole.q_learning.components.epsilon_greedy import EpsilonGreedy
from agents.cart_pole.q_learning.components.replay_buffer import ReplayBuffer
from agents.cart_pole.q_learning.deep_q_agent import DeepQAgent as CartDeepQAgent
from agents.plotting.training_history import TrainingHistory
from agents.virtual_gpu import VirtualGPU


class DeepQAgent(CartDeepQAgent):
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
