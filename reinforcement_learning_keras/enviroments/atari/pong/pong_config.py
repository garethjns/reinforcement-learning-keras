import os
from typing import Dict, Any

from reinforcement_learning_keras.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from reinforcement_learning_keras.agents.models.conv_nn import ConvNN
from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from reinforcement_learning_keras.enviroments.atari.atari_default_config import AtariDefaultConfig


class PongConfig(AtariDefaultConfig):
    """Defines configs tweaks for Pong."""
    env_spec = 'PongNoFrameskip-v4'

    def _build_for_dqn(self) -> Dict[str, Any]:
        return {'name': os.path.join(self.folder, 'DeepQAgent'),
                'env_spec': self.env_spec,
                'env_wrappers': self.env_wrappers,
                'model_architecture': ConvNN(observation_shape=(84, 84, self.frame_depth), n_actions=6,
                                             output_activation=None, opt='adam', learning_rate=0.000105),
                'gamma': 0.99,
                'final_reward': None,
                'eps': EpsilonGreedy(eps_initial=1.1, decay=0.000025, eps_min=0.01, decay_schedule='linear'),
                'replay_buffer': ContinuousBuffer(buffer_size=10000),
                'replay_buffer_samples': 32}


if __name__ == "__main__":
    EpsilonGreedy(eps_initial=1.1, decay=0.000025, eps_min=0.01, decay_schedule='linear',
                  perturb_increase_mag=0.4, perturb_increase_every=30000).simulate(plot=True, steps=100000)
    EpsilonGreedy(eps_initial=1.1, decay=0.00001, eps_min=0.01, decay_schedule='linear').simulate(plot=True,
                                                                                                  steps=100000)
