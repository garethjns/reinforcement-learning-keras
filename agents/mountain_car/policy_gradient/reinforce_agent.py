from agents.cart_pole.policy_gradient.reinforce_agent import ReinforceAgent as CartReinforceAgent
from agents.virtual_gpu import VirtualGPU


class ReinforceAgent(CartReinforceAgent):
    env_spec: str = "MountainCar-v0"
    learning_rate: float = 0.001

    @staticmethod
    def _final_reward(reward: float) -> float:
        return 250

    @classmethod
    def example(cls, n_episodes: int = 1000, render: bool = True) -> "ReinforceAgent":
        """Run a quick example with n_episodes and otherwise default settings."""
        VirtualGPU(128)
        agent = cls("MountainCar-v0")
        agent.train(verbose=True, render=render,
                    update_every=1,
                    max_episode_steps=1000,
                    n_episodes=n_episodes)

        return agent


if __name__ == "__main__":
    ReinforceAgent.example(render=True)
    