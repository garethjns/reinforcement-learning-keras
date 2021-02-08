# Based on https://keras.io/examples/rl/actor_critic_cartpole/

import os
from typing import Dict, Any, Tuple, Iterable, Callable, Optional, List, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.losses import huber_loss

from rlk.agents.agent_base import AgentBase
from rlk.agents.components.helpers.env_builder import EnvBuilder
from rlk.agents.components.history.training_history import TrainingHistory
from rlk.agents.components.replay_buffers.episode_tensor_buffer import EpisodeTensorBuffer
from rlk.agents.models.dense_nn_simple import DenseNNSimple
from rlk.agents.models.model_base import ModelBase


class ActorCriticAgent(AgentBase):
    _ac_model: Union[None, keras.Model]

    def __init__(self, training_history: TrainingHistory, model_architecture: ModelBase,
                 env_spec: str = "CartPole-v0", env_wrappers: Iterable[Callable] = (),
                 env_kwargs: Optional[Dict[str, Any]] = None, env_builder_kwargs: Optional[Dict[str, Any]] = None,
                 name: str = 'ACAgent', gamma: float = 0.99):

        if env_builder_kwargs is None:
            env_builder_kwargs = {}

        if env_kwargs is None:
            env_kwargs = {}

        self.training_history = training_history
        self.model_architecture = model_architecture
        self.env_spec = env_spec
        self.env_wrappers = env_wrappers
        self.env_kwargs = env_kwargs
        self.env_builder_kwargs = env_builder_kwargs
        self.name = name
        self.gamma = gamma

        self._buffer = EpisodeTensorBuffer()
        self.env_builder = EnvBuilder(env_spec=self.env_spec, env_wrappers=self.env_wrappers,
                                      env_kwargs=self.env_kwargs, **self.env_builder_kwargs)

        self._build_model()
        self._fn = f"{self.name}_{self.env_spec}"
        self.ready = True

    def __getstate__(self) -> Dict[str, Any]:
        return self._pickle_compatible_getstate()

    def get_weights(self) -> List[np.ndarray]:
        return self._ac_model.get_weights()

    def set_weights(self, weights: List[np.ndarray]) -> None:
        self._ac_model.set_weights(weights)

    def _save_models_and_buffer(self) -> None:
        if not os.path.exists(f"{self._fn}"):
            os.mkdir(f"{self._fn}")

        self._ac_model.save(f"{self._fn}/ac_model")

    def _load_models_and_buffer(self) -> None:
        self._target_model = keras.models.load_model(f"{self._fn}/ac_model")

    def unready(self) -> None:
        if self.ready:
            self._save_models_and_buffer()
            self._ac_model = None
            self._buffer.clear()
            keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
        super().unready()

    def check_ready(self):

        if not self.ready:
            self._load_models_and_buffer()

            super().check_ready()

    def get_action(self, s: Any, return_all=False) -> Tuple[int, Any]:
        """
        Get action to take from model.

        Note using: Model(state) instead of Model().predict() as this is called in tf.GradientTape() context during
        training and .predict() fails with "LookupError: No gradient defined for operation 'IteratorGetNext'
        (op type: IteratorGetNext)"

        :param s: State to predict for.
        :param return_all: If True, also return the action_probs and critic value, which are required for backprop
                           during training. If false, just return the action to take. Default False.
        """

        action_probs, critic_value = self._ac_model(self.transform(s))
        action = np.random.choice(range(self._ac_model.output[0].shape[1]), p=np.squeeze(action_probs))

        if return_all:
            return action, (action_probs, critic_value)
        else:
            return action

    def transform(self, s: Union[np.ndarray, List[np.ndarray], tf.Tensor]) -> Union[np.ndarray, List[np.ndarray]]:
        """Check shape of inputs, add Row dimension if required."""

        single_input = False
        model_inputs = self._ac_model.input
        if isinstance(model_inputs, tf.Tensor):
            # Input is a single array
            s = [s]
            model_inputs = [model_inputs]
            single_input = True

        s_trans = []
        for input_i, expected_input in zip(s, model_inputs):
            if len(input_i.shape) < len(expected_input.shape):
                # Add the None/row dimension
                s_trans.append(input_i[None, ...])
            else:
                # Leave as is
                s_trans.append(input_i)

        if single_input:
            return s_trans[0]
        else:
            return s_trans

    def _play_episode(self, max_episode_steps: int = 500, training: bool = False,
                      render: bool = True) -> Tuple[float, int]:
        """
        Play an episode while tracking gradients.

        This model will be updated using opt.apply_gradients(), rather than model.fit(). The model outputs are kept as
        tensors and saved to the replay buffer (along with reward) at each episode step. After the episode, the
        discounted returns are calculated, along with the loss, which still inside the tf.GradientTape() context.
        """

        self.env._max_episode_steps = max_episode_steps
        state = self.env.reset()
        total_reward = 0
        with tf.GradientTape(persistent=False) as self._last_tape:
            for frame in range(max_episode_steps):

                action, (action_probs, critic_value) = self.get_action(state, return_all=True)
                action_prob_log = tf.math.log(action_probs[0, action])

                state, reward, done, _ = self.env.step(action)
                self._buffer.append(action_prob=action_prob_log, reward=reward, critic_value=critic_value[0, 0])

                total_reward += reward

                if done:
                    break

            if training:
                self.update_model()

        return total_reward, frame

    def _calc_losses(self) -> Tuple[List[float], List[float]]:
        """Calculate the actor critic losses from the discounted rewards."""

        disc_rr: List[float] = self._buffer.get_discounted_rewards(gamma=self.gamma).squeeze()
        action_probs: List[tf.Tensor] = self._buffer.get_action_probs()
        critic_values: List[tf.Tensor] = self._buffer.get_critic_values()

        actor_losses = []
        critic_losses = []
        for ap, cv, drr in zip(action_probs, critic_values, disc_rr):
            # Difference between the (discounted) reward we got and the value the critic estimated
            diff = drr - cv
            # The action that led to this reward had a certain probability of being selected (ap here). Weight the
            # difference by probability to get the actor loss:
            actor_loss = -ap * diff

            # The critic loss is also based on the difference between the observed reward and estimated reward. Here
            # using Huber loss, which is robust to outliers
            # cv is a tensor here, drr is not.
            critic_loss = huber_loss(cv[None, ...], np.array(drr)[None, ...])

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        return actor_losses, critic_losses

    def update_model(self, *args, **kwargs) -> None:
        """
        Update model weights.

        Calculate losses, then run manual backprop.
        """
        actor_losses, critic_losses = self._calc_losses()
        self._backprop(actor_losses, critic_losses)

    def _backprop(self, actor_losses, critic_losses) -> None:
        """
        Get losses, calculate gradients from current GradientTape context, apply updates to model.
        """
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = self._last_tape.gradient(loss_value, self._ac_model.trainable_variables)
        self._ac_model.optimizer.apply_gradients(zip(grads, self._ac_model.trainable_variables))

    def _after_episode_update(self) -> None:
        self._buffer.clear()

    @classmethod
    def example(cls, config: Dict[str, Any]) -> "AgentBase":
        """TODO. See __main__ for now."""
        pass

    def _build_model(self) -> None:
        """
        Builder the actor and critic models.

        Most weights are shared between these two models, so efficient to use one model with two outputs:
         - Action probs (n_actions, )
         - Critic value (1, )
        """
        self._ac_model = self.model_architecture.compile(model_name='ac_model')


if __name__ == "__main__":
    ac_agent = ActorCriticAgent(env_spec='CartPole-v0',
                                training_history=TrainingHistory(plotting_on=True, plot_every=100, rolling_average=50),
                                model_architecture=DenseNNSimple(output_type='ac', observation_shape=(4,), n_actions=2,
                                                                 learning_rate=0.001, unit_scale=8,
                                                                 hidden_layer_activations='relu', opt='adam',
                                                                 output_activation='softmax'),
                                gamma=0.99)
    try:
        ac_agent.model_architecture.plot('rlk_model')
    except ImportError:
        # Probably missing pydot or graphviz
        pass

    ac_agent.train(verbose=True, render=False, n_episodes=10000, max_episode_steps=1000, checkpoint_every=0)
