from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union, Callable, Iterable

import numpy as np
from sklearn.linear_model import SGDRegressor

from reinforcement_learning_keras.agents.agent_base import AgentBase
from reinforcement_learning_keras.agents.components.helpers.env_builder import EnvBuilder
from reinforcement_learning_keras.agents.components.history.training_history import TrainingHistory
from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from reinforcement_learning_keras.enviroments.config_base import ConfigBase


@dataclass
class LinearQAgent(AgentBase):
    eps: EpsilonGreedy
    training_history: TrainingHistory
    env_spec: str = "CartPole-v0"
    env_wrappers: Iterable[Callable] = ()
    name: str = 'LinearQAgent'
    gamma: float = 0.99
    log_exemplar_space: bool = False
    final_reward: Union[float, None] = None

    def __post_init__(self, ) -> None:
        self.env_builder = EnvBuilder(env_spec=self.env_spec, env_wrappers=self.env_wrappers,
                                      env_kwargs=self.env_kwargs)
        self._build_model()

    def __getstate__(self) -> Dict[str, Any]:
        return self._pickle_compatible_getstate()

    def _build_model(self) -> None:
        """
        Build the models to estimate Q(a|s).

        Because this is linear regression with multiple outputs, use multiple models.
        """

        # Create SGDRegressor for each action in space and initialise by calling .partial_fit for the first time on
        # dummy data
        # Prep pipeline with scaler and rbfs paths

        mods = {a: SGDRegressor() for a in range(self.env.action_space.n)}
        for mod in mods.values():
            mod.partial_fit(self.transform(self.env.reset()), [0])

        self.mods = mods

    def transform(self, s: np.ndarray) -> np.ndarray:
        """Check shape. It's always single input with this agent."""
        if len(s.shape) == 1:
            s = s.reshape(1, -1)

        return s

    def partial_fit(self, s: np.ndarray, a: int, g: float) -> None:
        """
        Run partial fit for a single row of training data.

        :param s: The raw state observation.
        :param a: The action taken.
        :param g: The reward + discounted value of next state.
        """

        x = self.transform(s)
        self.mods[a].partial_fit(x, [g])

    def predict(self, s: np.ndarray) -> Dict[int, float]:
        """
        Given a single state observation, predict action values from each model.

        :param s: The raw state observation.
        :return: Dict containing action values, indexed by action id.
        """

        s = self.transform(s)

        return {a: float(mod.predict(s)) for a, mod in self.mods.items()}

    def get_best_action(self, s: np.ndarray) -> int:
        """
        Find the best action from the values predicted by each model.

        Remember preds is Dict[int, float]: This is essentially argmax on a dict.

        :param s: The raw state observation.
        :return: The action with the highest predicted value.
        """
        preds = self.predict(s)

        best_action_value = -np.inf
        best_action = 0
        for k, v in preds.items():
            if v > best_action_value:
                best_action_value = v
                best_action = k

        return best_action

    def get_action(self, s: np.ndarray, training: bool = False) -> int:
        """
        Get an action using epsilon greedy.

        Epsilon decays every time a random action is chosen.

        :param s: The raw state observation.
        :param training: Bool to indicate whether or not to use this experience to update the model. If False, just
                         returns best action.
        :return: The selected action.
        """
        action = self.eps.select(greedy_option=lambda: self.get_best_action(s),
                                 random_option=lambda: self.env.action_space.sample(),
                                 training=training)

        return action

    def update_model(self, s: np.ndarray, a: int, r: float, d: bool, s_: np.ndarray) -> None:
        """
        For a single step set, calculate discounted reward and update the appropriate action model.

        :param s: The raw state observation the action was selected for.
        :param a: The selected action.
        :param r: The reward for performing that action.
        :param d: Flag indicating if done.
        :param s_: The next state following the action.
        """
        # Update model
        if d:
            # Done flag is only true if env ends due to agent failure (not if max steps are reached). Punish.
            g = self.final_reward if self.final_reward is not None else 0
        else:
            # Calculate the reward for this step and the discounted max value of actions in the next state.
            g = r + self.gamma * np.max(list(self.predict(s_).values()))

        # Update the model s = x, g = y, and a is the model to update
        self.partial_fit(s, a, g)

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
        obs = self.env.reset()
        total_reward = 0
        for frame in range(max_episode_steps):
            action = self.get_action(obs, training=training)
            prev_obs = obs
            obs, reward, done, info = self.env.step(action)
            total_reward += reward

            if render:
                self.env.render()

            if training:
                self.update_model(s=prev_obs, a=action, r=reward, d=done, s_=obs)

            if done:
                break

        return total_reward, frame

    @classmethod
    def example(cls, config: ConfigBase, render: bool = True,
                n_episodes: int = 10, max_episode_steps: int = 500, update_every: int = 10) -> "AgentBase":
        """Create, train, and save agent for a given config."""
        config_dict = config.build()

        agent = cls(**config_dict)

        agent.train(verbose=True, render=render,
                    n_episodes=n_episodes, max_episode_steps=max_episode_steps, update_every=update_every)
        agent.save()

        return agent


if __name__ == "__main__":
    from reinforcement_learning_keras.enviroments.cart_pole import CartPoleConfig
    from reinforcement_learning_keras.enviroments import MountainCarConfig

    agent_cart_pole = LinearQAgent.example(CartPoleConfig(agent_type='linear_q', plot_during_training=True))
    agent_mountain_car = LinearQAgent.example(MountainCarConfig(agent_type='linear_q', plot_during_training=True))
