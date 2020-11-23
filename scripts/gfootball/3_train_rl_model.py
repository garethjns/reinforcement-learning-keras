"""
This script copies the weights from the pretrained model to a DQN agent and begins RL training.

It uses the open rules bot (see rlk.environments.gfootball.bots.open_rules_bot) as an additional policy for experience
collection. Note that this is done in a similar way to EpsilonGreedy (rather than the bot playing on it's own). On each
step, depending on epsilon, the agent will either sample an action from its own model, or from the bot
(rather than a totally random action)

Communication between the rlk env and the bot (which uses the Kaggle completion api and expects raw observations rather
than, eg. simple115 from the gym-wrapped env, is handled using the SimpleAndRawObsWrapper. This returns the simple115
observations to rl agent as normal, but additionally dumps the raw observations to disk for the bot to use.

There's also a frame buffer wrapper that stacks 2 frames.
"""

import os
from functools import partial

from tensorflow import keras
from tensorflow.python.keras.engine.functional import Functional

from rlk.agents.components.history.training_history import TrainingHistory
from rlk.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from rlk.agents.models.denser_nn import DenserNN
from rlk.agents.q_learning.deep_q_agent import DeepQAgent
from rlk.agents.q_learning.exploration.epsilon_policy import EpsilonPolicy
from rlk.environments.atari.environment_processing.frame_buffer_wrapper import FrameBufferWrapper
from rlk.environments.gfootball.bots.bot_config import BotConfig
from rlk.environments.gfootball.bots.open_rules_bot import agent as bot
from rlk.environments.gfootball.environment_processing.simple_and_raw_obs_wrapper import SimpleAndRawObsWrapper
from rlk.environments.gfootball.register_environments import register_all

N_EPISODES = 10


def copy_pretrained_model_weights(from_model: Functional, to_model: Functional) -> Functional:
    layer_names = [l.name for l in from_model.layers if l.name not in ['input']]
    for l_name in layer_names:
        to_model.get_layer(l_name).set_weights(from_model.get_layer(l_name).get_weights())

    return to_model


if __name__ == "__main__":

    register_all()

    FN = "nn_s115_pretrained_model"
    if os.path.exists(FN):
        path = FN
    else:
        path = f"../{FN}"
    pretrained_mod = keras.models.load_model(path)

    name = f'DQNAgent_nn_s115{FN}'
    agent = DeepQAgent(name=name,
                       env_spec="GFootball-kaggle_11_vs_11_001-v0",
                       gamma=0.99,
                       env_wrappers=[
                           # Wrapper to get s115 and dump raw observations to disk
                           partial(SimpleAndRawObsWrapper, raw_using=[],
                                   raw_dump_path=BotConfig().json_dump_path),
                           # Wrapper to buffer observations
                           partial(FrameBufferWrapper, obs_shape=(115,),
                                   buffer_length=2,
                                   buffer_function='stack')],
                       model_architecture=DenserNN(observation_shape=(115, 2), n_actions=19, dueling=False,
                                                   output_activation=None, opt='adam', learning_rate=0.00009),
                       eps=EpsilonPolicy(eps_initial=0.75, decay=0.000001, eps_min=0.01, policy=bot),
                       replay_buffer=ContinuousBuffer(buffer_size=8000),
                       training_history=TrainingHistory(plotting_on=True, plot_every=10,
                                                        agent_name=name))

    copy_pretrained_model_weights(from_model=pretrained_mod, to_model=agent._action_model)
    copy_pretrained_model_weights(from_model=pretrained_mod, to_model=agent._target_model)

    agent.train(n_episodes=N_EPISODES, render=False, checkpoint_every=100, max_episode_steps=3000)
    agent.save()

    # Also save a copy of just the model weights to use in evaluation steps.
    agent._action_model.save(f"nn_s115_rl_model")
