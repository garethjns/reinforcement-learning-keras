from dataclasses import dataclass
from typing import Dict, Any, Union, Tuple, Iterable, Callable

import numpy as np

from agents.agent_base import AgentBase
from agents.components.helpers.env_builder import EnvBuilder
from agents.components.helpers.virtual_gpu import VirtualGPU
from agents.components.history.training_history import TrainingHistory
from agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from enviroments.config_base import ConfigBase
from enviroments.model_base import ModelBase


@dataclass()
class DeepQAgent(AgentBase):
    replay_buffer: ContinuousBuffer
    eps: EpsilonGreedy
    training_history: TrainingHistory
    model_architecture: ModelBase
    env_spec: str = "CartPole-v0"
    env_wrappers: Iterable[Callable] = ()
    name: str = 'DQNAgent'
    gamma: float = 0.99
    frame_depth: int = 1
    replay_buffer_samples: int = 75
    learning_rate: float = 0.001
    final_reward: Union[float, None] = None

    _action_model_weights: Union[np.ndarray, None] = None

    def __post_init__(self) -> None:
        self.env_builder = EnvBuilder(self.env_spec, self.env_wrappers)
        self._build_model()

    def __getstate__(self) -> Dict[str, Any]:
        return self._pickle_compatible_getstate()

    def unready(self) -> None:
        if self._action_model is not None:
            self._action_model_weights = self._action_model.get_weights()
            self._action_model = None
            self._value_model = None

    def check_ready(self):
        super().check_ready()
        if self._action_model is None:
            self._build_model()

    def _build_model(self) -> None:
        """
        Prepare two of the same model.

        The action model is used to pick actions and the value model is used to predict value of Q(s', a). Action model
        weights are updated on every buffer sample + training step. The value model is never directly trained, but it's
        weights are updated to match the action model at the end of each episode.

        :return:
        """
        self.model_architecture.compile(model_name='action_model')

        self._action_model = self.model_architecture.compile(model_name='action_model')
        self._value_model = self.model_architecture.compile(model_name='value_model')

        # If existing model weights have been passed at object instantiation, apply these. This is likely will only
        # be done when unpickling or when preparing to pickle this object.
        if self._action_model_weights is not None:
            self._action_model.set_weights(self._action_model_weights)
            self._value_model.set_weights(self._action_model_weights)
            self._action_model_weights = None

    def transform(self, s: np.ndarray) -> np.ndarray:
        """Check input shape, add Row dimension if required."""

        if len(s.shape) < len(self._action_model.input.shape):
            s = np.expand_dims(s, 0)

        return s

    def update_experience(self, s: np.ndarray, a: int, r: float, d: bool) -> None:
        """
        First the most recent step is added to the buffer.

        Note that s' isn't saved because there's no need. It'll be added next step. s' for any s is always index + 1 in
        the buffer.
        """

        # Add s, a, r, d to experience buffer
        self.replay_buffer.append((s, a, r, d))

    def update_model(self) -> None:
        """
        Sample a batch from the replay buffer, calculate targets using value model, and train action model.

        If the buffer is below its minimum size, no training is done.

        If the buffer has reached its minimum size, a training batch from the replay buffer and the action model is
        updated.

        This update samples random (s, a, r, s') sets from the buffer and calculates the discounted reward for each set.
        The value of the actions at states s and s' are predicted from the value model. The action model is updated
        using these value predictions as the targets. The value of performed action is updated with the discounted
        reward (using its value prediction at s'). ie. x=s, y=[action value 1, action value 2].

        The predictions from the value model s, s', and the update of the action model is done in batch before and
        after the loop. The loop then iterates over the rows. Note that an alternative is doing the prediction and
        fit calls on singles rows in the loop. This would be very inefficient, especially if using a GPU.
        """

        # If buffer isn't full, don't train
        if not self.replay_buffer.full:
            return

        # Else sample batch from buffer
        ss, aa, rr, dd, ss_ = self.replay_buffer.sample_batch(self.replay_buffer_samples)

        # For each sample, calculate targets using Bellman eq and value/target network

        ss = np.array(ss)
        y_now = self._value_model.predict(ss)
        y_future = self._value_model.predict(np.array(ss_))
        y = []
        # TODO: Vectorize loop (although predict/train already outside so won't be major perf benefit)
        for i, (state, action, reward, done, state_) in enumerate(zip(ss, aa, rr, dd, ss_)):
            if done:
                # If done, reward is just this step. For cart pole can only be done if agent has failed, so punish.
                g = self.final_reward if self.final_reward is not None else reward
            else:
                # Otherwise, it's the reward plus the predicted max value of next action
                g = reward + self.gamma * np.max(y_future[i, :])

            # Set non-acted actions to y_now preds and acted action to y_future pred
            y_ = y_now[i, :]
            y_[action] = g

            y.append(y_)

        # Fit action
        self._action_model.train_on_batch(ss, np.stack(y))

    def get_best_action(self, s: np.ndarray) -> np.ndarray:
        """
        Get best action(s) from model - the one with the highest predicted value.
        :param s: A single or multiple rows of state observations.
        :return: The selected action.
        """

        preds = self._action_model.predict(self.transform(s))

        return np.argmax(preds)

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

    def update_value_model(self) -> None:
        """
        Update the value model with the weights of the action model (which is updated each step).

        The value model is updated less often to aid stability.
        """
        self._value_model.set_weights(self._action_model.get_weights())

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
                self.update_experience(s=prev_obs, a=action, r=reward, d=done)
                # Action model is updated in TD(λ) fashion
                self.update_model()

            if done:
                break

        return total_reward, frame

    def _after_episode_update(self) -> None:
        """Value model synced with action model at the end of each episode."""
        self.update_value_model()

    @classmethod
    def example(cls, config: ConfigBase, render: bool = True,
                n_episodes: int = 500, max_episode_steps: int = 500, update_every: int = 10) -> "DeepQAgent":
        """Create, train, and save agent for a given config."""
        VirtualGPU(config.gpu_memory)
        config_dict = config.build()

        agent = cls(**config_dict)

        agent.train(verbose=True, render=render,
                    n_episodes=n_episodes, max_episode_steps=max_episode_steps, update_every=update_every,
                    checkpoint_every=100)
        agent.save(f"{agent.name}_{config_dict['env_spec']}.pkl")

        return agent


if __name__ == "__main__":
    from enviroments.pong.pong_config import PongConfig
    from enviroments.cart_pole.cart_pole_config import CartPoleConfig
    from enviroments.mountain_car.mountain_car_config import MountainCarConfig

    # DQNs
    agent_cart_pole = DeepQAgent.example(CartPoleConfig(agent_type='dqn', plot_during_training=True))
    agent_mountain_car = DeepQAgent.example(MountainCarConfig(agent_type='dqn', plot_during_training=True))
    agent_pong = DeepQAgent.example(PongConfig(agent_type='dqn', plot_during_training=True),
                                    max_episode_steps=10000, update_every=5)

    # Dueling DQNs
    dueling_agent_cart_pole = DeepQAgent.example(CartPoleConfig(agent_type='dueling_dqn', plot_during_training=True))
    dueling_agent_mountain_car = DeepQAgent.example(MountainCarConfig(agent_type='dueling_dqn',
                                                                      plot_during_training=True))
