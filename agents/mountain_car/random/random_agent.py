from agents.cart_pole.random.random_agent import RandomAgent as CartRandomAgent


class RandomAgent(CartRandomAgent):
    env_spec: str = "MountainCar-v0"

    @classmethod
    def example(cls, n_episodes: int = 1000, render: bool = True) -> "RandomAgent":
        agent = cls("MountainCar-v0")
        agent.train(verbose=True, render=render,
                    n_episodes=n_episodes,
                    max_episode_steps=1000,
                    checkpoint_every=False)

        return agent


if __name__ == "__main__":
    RandomAgent.example()
