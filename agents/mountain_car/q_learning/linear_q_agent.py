from dataclasses import dataclass

from agents.cart_pole.q_learning.linear_q_agent import LinearQAgent as CartLinearQAgent


@dataclass
class LinearQAgent(CartLinearQAgent):
    env_spec: str = "MountainCar-v0"

    @staticmethod
    def _final_reward(reward: float) -> float:
        return 500

    @classmethod
    def example(cls, n_episodes: int = 1000, render: bool = True) -> "LinearQAgent":
        agent = cls()
        agent.train(verbose=True, render=render,
                    max_episode_steps=2000,
                    n_episodes=n_episodes,
                    checkpoint_every=25)

        return agent


if __name__ == "__main__":
    agent_ = LinearQAgent.example()
    agent_.save('linear_q_mountain_car_example.pkl')
