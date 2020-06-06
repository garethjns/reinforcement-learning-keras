from agents.cart_pole.q_learning.linear_q_agent import LinearQAgent as CartLinearQAgent


class LinearQAgent(CartLinearQAgent):
    env_spec: str = "MountainCar-v0"

    @staticmethod
    def _final_reward(reward: float) -> float:
        return 250

    @classmethod
    def example(cls, n_episodes: int = 1000, render: bool = True) -> "LinearQAgent":
        agent = cls("MountainCar-v0")
        agent.train(verbose=True, render=render,
                    max_episode_steps=1000,
                    n_episodes=n_episodes,
                    checkpoint_every=10)

        return agent


if __name__ == "__main__":
    LinearQAgent.example()
