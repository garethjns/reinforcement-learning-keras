from agents.cart_pole.random.random_agent import RandomAgent as CartRandomAgent

from dataclasses import dataclass


@dataclass
class RandomAgent(CartRandomAgent):
    env_spec: str = "Pong-v0"

    @classmethod
    def example(cls, n_episodes: int = 1000, render: bool = True) -> "RandomAgent":
        agent = cls()
        agent.train(verbose=True, render=render,
                    n_episodes=n_episodes,
                    max_episode_steps=2000,
                    checkpoint_every=False)

        return agent


if __name__ == "__main__":
    RandomAgent.example()
