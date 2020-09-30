from dataclasses import dataclass
from typing import Any, Dict, Tuple, Iterable, Callable

from reinforcement_learning_keras.agents.agent_base import AgentBase
from reinforcement_learning_keras.agents.components.helpers.env_builder import EnvBuilder
from reinforcement_learning_keras.agents.components.history.training_history import TrainingHistory
from reinforcement_learning_keras.agents.random.random_model import RandomModel
from reinforcement_learning_keras.enviroments.config_base import ConfigBase


@dataclass
class RandomAgent(AgentBase):
    """
    A non-abstract agent implementing AgentBase interface but acts randomly and learns nothing.

    Useful as a baseline and for testing.
    """
    env_spec: str
    training_history: TrainingHistory
    env_wrappers: Iterable[Callable] = ()
    name: str = 'RandomAgent'

    def __post_init__(self) -> None:
        self.env_builder = EnvBuilder(env_spec=self.env_spec, env_wrappers=self.env_wrappers,
                                      env_kwargs=self.env_kwargs)
        self._build_model()

    def __getstate__(self) -> Dict[str, Any]:
        return self._pickle_compatible_getstate()

    def _build_model(self) -> None:
        """Set model function. Note using a lambda breaks pickle support."""
        self.model = RandomModel(self.env.action_space.n)

    def update_model(self, *args, **kwargs) -> None:
        """No model to update."""
        pass

    def get_action(self, s: Any, **kwargs) -> int:
        return self.model.predict()

    def _play_episode(self, max_episode_steps: int = 500,
                      training: bool = False, render: bool = True) -> Tuple[float, int]:
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
        for frame in range(max_episode_steps):
            action = self.get_action(None)
            _, reward, done, _ = self.env.step(action)
            total_reward += reward

            if render:
                self.env.render()

            if done:
                break

        return total_reward, frame

    @classmethod
    def example(cls, config: ConfigBase, render: bool = True,
                n_episodes: int = 500, max_episode_steps: int = 500) -> "RandomAgent":
        """Create, train, and save agent for a given config."""
        config_dict = config.build()

        agent = cls(**config_dict)

        agent.train(verbose=True, render=render,
                    n_episodes=n_episodes, max_episode_steps=max_episode_steps)
        agent.save()

        return agent


if __name__ == "__main__":
    from reinforcement_learning_keras.enviroments import PongConfig
    from reinforcement_learning_keras.enviroments.cart_pole import CartPoleConfig
    from reinforcement_learning_keras.enviroments import MountainCarConfig

    agent_mountain_car = RandomAgent.example(
        MountainCarConfig(agent_type='random', plot_during_training=True), max_episode_steps=1500, render=False)

    agent_cart_pole = RandomAgent.example(CartPoleConfig(agent_type='random', plot_during_training=True))
    agent_mountain_car = RandomAgent.example(MountainCarConfig(agent_type='random', plot_during_training=True))
    agent_pong = RandomAgent.example(PongConfig(agent_type='random', plot_during_training=True),
                                     max_episode_steps=10000)
