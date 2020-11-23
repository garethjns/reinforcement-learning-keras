import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, Union, Tuple, Iterable, Callable, List

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlk.agents.agent_base import AgentBase
from rlk.agents.components.helpers.env_builder import EnvBuilder
from rlk.agents.components.helpers.virtual_gpu import VirtualGPU
from rlk.agents.components.history.training_history import TrainingHistory
from rlk.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from rlk.agents.models.model_base import ModelBase
from rlk.agents.q_learning.exploration.epsilon_base import EpsilonBase
from rlk.environments.config_base import ConfigBase

tf.compat.v1.disable_eager_execution()


@dataclass
class DeepQAgent(AgentBase):
    replay_buffer: ContinuousBuffer
    eps: EpsilonBase
    training_history: TrainingHistory
    model_architecture: ModelBase
    double: bool = False
    noisy: bool = False
    env_spec: str = "CartPole-v0"
    env_wrappers: Iterable[Callable] = ()
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    env_builder_kwargs: Dict[str, Any] = field(default_factory=dict)
    name: str = 'DQNAgent'
    gamma: float = 0.99
    replay_buffer_samples: int = 75
    final_reward: Union[float, None] = None

    def __post_init__(self) -> None:
        if self.env_builder_kwargs is None:
            self.env_builder_kwargs = {}

        self.env_builder = EnvBuilder(env_spec=self.env_spec, env_wrappers=self.env_wrappers,
                                      env_kwargs=self.env_kwargs, **self.env_builder_kwargs)

        self._build_model()
        self._fn = f"{self.name}_{self.env_spec}"
        self.ready = True

    def __getstate__(self) -> Dict[str, Any]:
        return self._pickle_compatible_getstate()

    def _save_models_and_buffer(self) -> None:
        if not os.path.exists(f"{self._fn}"):
            os.mkdir(f"{self._fn}")

        self._action_model.save(f"{self._fn}/action_model")
        self._target_model.save(f"{self._fn}/target_model")
        self.replay_buffer.save(f"{self._fn}/replay_buffer.joblib")

    def _load_models_and_buffer(self) -> None:
        self._action_model = keras.models.load_model(f"{self._fn}/action_model")
        self._target_model = keras.models.load_model(f"{self._fn}/target_model")
        self.replay_buffer = ContinuousBuffer.load(f"{self._fn}/replay_buffer.joblib")

    def get_weights(self) -> np.ndarray:
        return self._action_model.get_weights()

    def set_weights(self, weights: np.ndarray) -> None:
        self._action_model.set_weights(weights)
        self._target_model.set_weights(weights)

    def unready(self) -> None:
        if self.ready:
            self._save_models_and_buffer()
            self._action_model = None
            self._target_model = None
            self.replay_buffer = None
            keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
        super().unready()

    def check_ready(self):

        if not self.ready:
            self._load_models_and_buffer()

            super().check_ready()

    def _build_model(self) -> None:
        """
        Prepare two of the same model.

        The action model is used to pick actions and the value model is used to predict value of Q(s', a). Action model
        weights are updated on every buffer sample + training step. The target model is never directly trained, but it's
        weights are updated to match the action model at the end of each episode.

        :return:
        """
        self._action_model = self.model_architecture.compile(model_name='action_model', loss='mse')
        self._target_model = self.model_architecture.compile(model_name='target_model', loss='mse')

    def transform(self, s: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Check shape of inputs, add Row dimension if required.
        """

        single_input = False
        model_inputs = self._action_model.input
        if isinstance(model_inputs, tf.Tensor):
            # Input is a single array
            s = [s]
            model_inputs = [model_inputs]
            single_input = True

        s_trans = []
        for input_i, expected_input in zip(s, model_inputs):
            if len(input_i.shape) < len(expected_input.shape):
                # Add the None/row dimension
                s_trans.append(np.expand_dims(input_i, 0))
            else:
                # Leave as is
                s_trans.append(input_i)

        if single_input:
            return s_trans[0]
        else:
            return s_trans

    def update_experience(self, s: np.ndarray, a: int, r: float, d: bool) -> None:
        """
        First the most recent step is added to the buffer.

        Note that s' isn't saved because there's no need. It'll be added next step. s' for any s is always index + 1 in
        the buffer.
        """

        # Add s, a, r, d to experience buffer
        self.replay_buffer.append((s, a, r, d))

    @staticmethod
    def _stack_inputs(ss: Union[np.ndarray, List[np.ndarray]],
                      ss_: Union[np.ndarray, List[np.ndarray]]) -> Tuple[Union[np.ndarray, List[np.ndarray]],
                                                                         Union[np.ndarray, List[np.ndarray]],
                                                                         Union[np.ndarray, List[np.ndarray]]]:
        """
        Convert ss and ss_ inputs to arrays, handling single array input or list of arrays.

        This is necessary for models with two separate array inputs - for example conv + dense, as np.array() alone
        will not work correctly.

        Returns both as arrays, and a stack of them combined.
        """

        # Check row of the inputs
        if isinstance(ss[0], np.ndarray):
            # Input is array, can have multiple dimensions
            ss_array = np.array(ss)
            ss__array = np.array(ss_)
            stack = np.vstack((ss_array, ss__array))

            return ss_array, ss__array, stack

        else:
            # Input presumably list (or tuple) of arrays. Stack separately.
            n_inputs = len(ss[0])
            stack_list, ss_list, ss__list = [], [], []
            for input_i in range(n_inputs):
                ss_array = np.array([ssr[input_i] for ssr in ss])
                ss__array = np.array([ss_r[input_i] for ss_r in ss_])
                ss_list.append(ss_array)
                ss__list.append(ss__array)
                stack_list.append(np.vstack((ss_array, ss__array)))

            return ss_list, ss__list, stack_list

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

        GPU Performance notes (with 1080ti and eps @ 0.01, while rendering pong):
          - Looping here with 2 predict calls and 1 train call (each single rows) is unusably slow.
          - Two predict calls before loop and 1 train call after (on batches) runs at ~16 fps for pong (~2 GPU util).
          - Switching TF to non-eager mode improves performance to 50fps (~7% GPU util) (also stops memory leaks).
          - Reducing the predict calls to 1 by joining s and s' increases performance to ~73 fps (~14% util).
            - Render off: ~81fps (~16% util)
          - Vectorizing out the remaining loop: ~73fps (~14% util)
            - Render off: ~84fps (~16% util)

        Ryzen with 1080ti (render off), 32 samples dqn with SplitterConvNN
         - Single
           - Diff buffer: 105 f/s (14%)
           - Stacked buffer:  61 f/s (22%)
         - Multi 3x
           - Diff buffer: 88 f/s (13%) x3 = 264 f/s (39%)
           - Stacked buffer: 46 f/s (16%) x3 = 138 f/s (48%)
        - Multi 6x
           - Diff buffer: 63 f/s (10%) x6 = 378 f/s (60%)
           - Stacked buffer: 28 f/s (10%) x6 = 168 f/s (60%)
        - Multi 12x (not CPU limited)
           - Diff buffer: 14 f/s (4-11% much more variable) x10 = 168 f/s (~60-70%)
           - Stacked buffer: 10 f/s (4-12% much more variable) x10 = 120 f/s (~60-70%)

        Ryzen with 1080ti (render off), 32 samples dqn withConvNN
         - Single
           - Diff buffer: 110 f/s (18%)
           - Stacked buffer: 84 f/s (19%)
         - Multi 3x
           - Diff buffer: 84 f/s (15%) x3 = 252 f/s (45%)
           - Stacked buffer: 68 f/s (15%) x3 = 204 f/s (45%)
        - Multi 6x
           - Diff buffer: 51 f/s (10%) x6 = 306 f/s (60%)
           - Stacked buffer: 44 f/s (10%) x6 = 264 f/s (60%)
        - Multi 12x (not CPU limited)
           - Diff buffer: 7 f/s (2-30%) x10 = 84 f/s (60-90%)
           - Stacked buffer: 9 f/s (%) x10 = 108 f/s (%)
        """

        # If buffer isn't full, don't train
        if not self.replay_buffer.full:
            return

        # Else sample batch from buffer
        ss, aa, rr, dd, ss_ = self.replay_buffer.sample_batch(self.replay_buffer_samples)

        # Calculate estimated S,A values for current states and next states. These are stacked together first to avoid
        # making two separate predict calls
        ss, ss_, ss_and_ss_ = self._stack_inputs(ss, ss_)
        y_now_and_future = self._target_model.predict_on_batch(ss_and_ss_)
        y_now = y_now_and_future[0:self.replay_buffer_samples]
        y_future = y_now_and_future[self.replay_buffer_samples::]

        # Update rewards where not done with y_future predictions
        dd_mask = np.array(dd, dtype=bool).squeeze()
        rr = np.array(rr, dtype=float).squeeze()

        # Gather max action indexes and update relevant actions in y
        if self.double:
            # If using double dqn select best actions using the action model, but the value of those action using the
            # target model (already have in y_future). Note that this doesn't appear to have as much of a performance
            # cost as might be expected - presumably because the ss_ data is already on the GPU so transfer delay
            # avoided.
            y_future_action_model = self._action_model.predict_on_batch(ss_)
            selected_actions = np.argmax(y_future_action_model[~dd_mask, :], axis=1)
        else:
            # If normal dqn select targets using target model, and value of those from target model too
            selected_actions = np.argmax(y_future[~dd_mask, :], axis=1)

        # Update reward values with estimated values (where not done) and final rewards (where done)
        rr[~dd_mask] += y_future[~dd_mask, selected_actions]
        if self.final_reward is not None:
            # If self.final_reward is set, set done cases to this value. Else leave as observed reward.
            rr[dd_mask] = self.final_reward
        aa = np.array(aa, dtype=int)
        np.put_along_axis(y_now, aa.reshape(-1, 1), rr.reshape(-1, 1), axis=1)

        # Fit model with updated y_now values
        self._action_model.train_on_batch(ss, y_now)

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
        action = self.eps.select(state=s, greedy_option=lambda: self.get_best_action(s),
                                 training=training)

        return action

    def update_target_model(self) -> None:
        """
        Update the value model with the weights of the action model (which is updated each step).

        The value model is updated less often to aid stability.
        """
        self._target_model.set_weights(self._action_model.get_weights())

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
        self.update_target_model()

    @classmethod
    def example(cls, config: ConfigBase, render: bool = True,
                n_episodes: int = 500, max_episode_steps: int = 500, update_every: int = 10,
                checkpoint_every: int = 100) -> "DeepQAgent":
        """For a given config, create new, or load existing agent. Then train and save agent."""

        VirtualGPU(config.gpu_memory)

        config_dict = config.build()
        if os.path.exists(config_dict['name']):
            agent = cls.load(config_dict['name'])
            warnings.warn('Loaded existing agent.')
        else:
            agent = cls(**config_dict)

        agent.train(verbose=True, render=render,
                    n_episodes=n_episodes, max_episode_steps=max_episode_steps, update_every=update_every,
                    checkpoint_every=checkpoint_every)
        agent.save()

        return agent

    def _save_self(self):
        """Save agent.joblib."""

        if not os.path.exists(f"{self._fn}"):
            os.mkdir(f"{self._fn}")
        joblib.dump(self, f"{self._fn}/agent.joblib")

    def save(self, make_ready: bool = True) -> None:
        """
        Saves buffer, etc. via unready and agent.joblib with save.

        :param make_ready: Make agent ready agent after saving (reload buffer, etc). This can be skipped to save time
                           in some situations. Default True.
        """

        self.unready()
        self._save_self()
        if make_ready:
            self.check_ready()

    @classmethod
    def load(cls, fn: str) -> "DeepQAgent":
        new_agent = joblib.load(f"{fn}/agent.joblib")
        new_agent.check_ready()

        return new_agent


if __name__ == "__main__":
    from rlk.environments.atari.pong.pong_config import PongConfig
    from rlk.environments.cart_pole import CartPoleConfig
    from rlk.environments import MountainCarConfig

    # DQNs
    agent_cart_pole = DeepQAgent.example(CartPoleConfig(agent_type='dqn', plot_during_training=True), render=False)
    agent_mountain_car = DeepQAgent.example(MountainCarConfig(agent_type='dqn', plot_during_training=True))
    agent_pong = DeepQAgent.example(PongConfig(agent_type='dqn', plot_during_training=True),
                                    max_episode_steps=10000, update_every=5, render=False, checkpoint_every=10)

    # Dueling DQNs
    dueling_agent_cart_pole = DeepQAgent.example(CartPoleConfig(agent_type='dueling_dqn', plot_during_training=True))
    dueling_agent_mountain_car = DeepQAgent.example(MountainCarConfig(agent_type='dueling_dqn',
                                                                      plot_during_training=True))
