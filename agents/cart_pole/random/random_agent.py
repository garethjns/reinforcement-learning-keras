from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from agents.agent_base import AgentBase
from agents.plotting.training_history import TrainingHistory


@dataclass
class RandomAgent(AgentBase):
    """
    A non-abstract agent implementing AgentBase interface but acts randomly and learns nothing.

    Useful as a baseline and for testing.
    """
    env_spec: str = "CartPole-v0"
    name: str = 'RandomAgent'
    plot_during_training: bool = True

    def __post_init__(self):
        self.history = TrainingHistory(plotting_on=self.plot_during_training,
                                       plot_every=25,
                                       rolling_average=12,
                                       agent_name=self.name)
        self._set_env()
        self._build_model()

    def __getstate__(self) -> Dict[str, Any]:
        return self._pickle_compatible_getstate()

    def _model_f(self):
        return int(np.random.randint(0, self.env.action_space.n, 1))

    def _build_model(self) -> None:
        """Set model function. Note using a lambda breaks pickle support."""
        self.model = self._model_f

    def update_model(self, *args, **kwargs) -> None:
        """No model to update."""
        pass

    def get_action(self, s: Any, **kwargs) -> int:
        return self.model()

    def play_episode(self, max_episode_steps: int = 500,
                     training: bool = False, render: bool = True) -> float:
        """
        Play a single episode and return the total reward.

        :param max_episode_steps: Max steps before stopping, overrides any time limit set by Gym.
        :param training: Bool to indicate whether or not to use this experience to update the model.
        :param render: Bool to indicate whether or not to call env.render() each training step.
        :return: The total real reward for the episode.
        """
        self.env._max_episode_steps = max_episode_steps
        _ = self.env.reset()
        total_reward = 0
        for _ in range(max_episode_steps):
            action = self.get_action(None)
            _, reward, done, _ = self.env.step(action)
            total_reward += reward

            if render:
                self.env.render()

            if done:
                break

        return total_reward

    @classmethod
    def example(cls, n_episodes: int = 1000, render: bool = True) -> "RandomAgent":
        agent = cls("CartPole-v0")
        agent.train(verbose=True, render=render,
                    n_episodes=n_episodes,
                    checkpoint_every=False)

        return agent


if __name__ == "__main__":
    RandomAgent.example()
