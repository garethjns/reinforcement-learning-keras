from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from agents.agent_base import AgentBase
from agents.cart_pole.environment_processing.clipper import Clipper
from agents.cart_pole.q_learning.components.epsilon_greedy import EpsilonGreedy
from agents.plotting.training_history import TrainingHistory


@dataclass
class LinearQAgent(AgentBase):
    env_spec: str = "CartPole-v0"
    name: str = 'LinearQAgent'
    eps: EpsilonGreedy = None
    gamma: float = 0.99
    plot_during_training: bool = True
    log_exemplar_space: bool = False

    def __post_init__(self, ) -> None:
        self.history = TrainingHistory(plotting_on=self.plot_during_training,
                                       plot_every=200,
                                       rolling_average=12,
                                       agent_name=self.name)

        if self.eps is None:
            # Prepare the default EpsilonGreedy sampler if one is not specified.
            self.eps = EpsilonGreedy(eps_initial=0.4,
                                     eps_min=0.01)

        self._set_env()
        self._build_pp()
        self._build_model()

    def _build_pp(self) -> None:
        """
        Create and fit the pre-processing pipeline.

        obs -> clip -> standard scaler -> RBF features

        The scalar is fitted using a training set generated by sampling from the environment many times.
        """

        # Prep pipeline with scaler and rbfs paths
        fu = FeatureUnion([('rbf0', RBFSampler(gamma=100, n_components=60)),
                           ('rbf1', RBFSampler(gamma=1, n_components=60)),
                           ('rbf2', RBFSampler(gamma=0.02, n_components=60))])

        pipe = Pipeline([('clip', Clipper()),
                         ('ss', StandardScaler()),
                         ('rbfs', fu)])

        # Sample observations from env and fit pipeline
        obs = np.array([self._env.observation_space.sample() for _ in range(10000)])
        if self.log_exemplar_space:
            obs = np.sign(obs) * np.log(np.abs(obs))
        pipe.fit(obs)

        self.pp = pipe

    def _build_model(self) -> None:
        """
        Build the models to estimate Q(a|s).

        Because this is linear regression with multiple outputs, use multiple models.
        """

        # Create SGDRegressor for each action in space and initialise by calling .partial_fit for the first time on
        # dummy data
        mods = {a: SGDRegressor() for a in range(self._env.action_space.n)}
        for mod in mods.values():
            mod.partial_fit(self.transform(self._env.reset()), [0])

        self.mods = mods

    def transform(self, s: np.ndarray) -> np.ndarray:
        """Run the processing pipeline on a single state."""
        if len(s.shape) == 1:
            s = s.reshape(1, -1)

        return self.pp.transform(s)

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
                                 random_option=lambda: self._env.action_space.sample(),
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
            g = - 200
        else:
            # Calculate the reward for this step and the discounted max value of actions in the next state.
            g = r + self.gamma * np.max(list(self.predict(s_).values()))

        # Update the model s = x, g = y, and a is the model to update
        self.partial_fit(s, a, g)

    def play_episode(self, max_episode_steps: int = 500,
                     training: bool = False, render: bool = True) -> float:
        """
        Play a single episode and return the total reward.

        :param max_episode_steps: Max steps before stopping, overrides any time limit set by Gym.
        :param training: Bool to indicate whether or not to use this experience to update the model.
        :param render: Bool to indicate whether or not to call env.render() each training step.
        :return: The total real reward for the episode.
        """
        self._env._max_episode_steps = max_episode_steps
        obs = self._env.reset()
        total_reward = 0
        for _ in range(max_episode_steps):
            action = self.get_action(obs, training=training)
            prev_obs = obs
            obs, reward, done, info = self._env.step(action)
            total_reward += reward

            if render:
                self._env.render()

            if training:
                self.update_model(s=prev_obs, a=action, r=reward, d=done, s_=obs)

            if done:
                break

        return total_reward

    @classmethod
    def example(cls, n_episodes: int = 1000, render: bool = True) -> "LinearQAgent":
        agent = cls("CartPole-v0")
        agent.train(verbose=True, render=render,
                    n_episodes=n_episodes)

        return agent


if __name__ == "__main__":
    LinearQAgent.example()
