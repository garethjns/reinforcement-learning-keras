import os
import warnings
from typing import Any, Dict

from rlk.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from rlk.agents.models.dense_nn import DenseNN
from rlk.agents.models.splitter_conv_nn import SplitterConvNN
from rlk.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from rlk.environments.config_base import ConfigBase
from rlk.environments.gfootball.environment_processing.gf_remote_wrapper import GFRemoteWrapper
from rlk.environments.gfootball.environment_processing.simple_and_smm_obs_wrapper import SimpleAndSMMObsWrapper
from rlk.environments.gfootball.environment_processing.smm_frame_process_wrapper import \
    SMMFrameProcessWrapper
from rlk.environments.gfootball.register_environments import register_all, SUPPORTED_ENVS

register_all()


class GFootballConfig(ConfigBase):
    supported_agents = ('linear_q', 'dqn', 'double_dqn', 'dueling_dqn', 'double_dueling_dqn')
    supported_envs = tuple(SUPPORTED_ENVS + ["GFootball-11_vs_11_kaggle-simple115v2-v0",
                                             "GFootball-11_vs_11_kaggle-SMM-v0"])
    gpu_memory: int = 2048

    def __init__(self, *args, env_spec: str = "GFootball-11_vs_11_kaggle-simple115v2-v0",
                 using_simple_obs: bool = True, using_smm_obs: bool = True,
                 remote: bool = False, **kwargs):
        if env_spec not in self.supported_envs:
            warnings.warn(f"Unknown env {env_spec}")
        self.env_spec = env_spec
        self.using_simple_obs = using_simple_obs
        self.using_smm_obs = using_smm_obs
        self.remote = remote
        super().__init__(*args, **kwargs)

    @property
    def _default_training_history_kwargs(self) -> Dict[str, Any]:
        return {"plotting_on": self.plot_during_training,
                "plot_every": 8, "rolling_average": 8}

    @staticmethod
    def _build_with_dense_model(dueling: bool = False) -> Dict[str, Any]:
        return {"model_architecture": DenseNN(observation_shape=(115,), n_actions=19, dueling=dueling,
                                              output_activation=None, opt='adam', learning_rate=0.000105)}

    @staticmethod
    def _build_with_splitter_conv_model(dueling: bool = False) -> Dict[str, Any]:
        return {"model_architecture": SplitterConvNN(observation_shape=(72, 96, 4), n_actions=19, dueling=dueling,
                                                     output_activation=None, opt='adam', learning_rate=0.000105),
                "env_wrappers": [SMMFrameProcessWrapper]}

    def _build_with_splitter_conv_and_dense_model(self, dueling: bool = False) -> Dict[str, Any]:
        return {"model_architecture": SplitterConvNN(observation_shape=(72, 96, 4), n_actions=19, dueling=dueling,
                                                     additional_dense_input_shape=(115,), output_activation=None,
                                                     opt='adam', learning_rate=0.000105),
                "env_wrappers": [GFRemoteWrapper] if self.remote else [SimpleAndSMMObsWrapper, SMMFrameProcessWrapper]}

    def _build_for_dqn(self, dueling: bool = False) -> Dict[str, Any]:
        if self.using_simple_obs & self.using_smm_obs:
            model_config = self._build_with_splitter_conv_and_dense_model(dueling)
        elif self.using_simple_obs:
            model_config = self._build_with_dense_model(dueling)
        elif self.using_smm_obs:
            model_config = self._build_with_splitter_conv_model(dueling)
        else:
            raise NotImplementedError()

        config = {'name': os.path.join(self.folder, 'DeepQAgent'),
                  'env_spec': self.env_spec,
                  'gamma': 0.992,
                  'final_reward': 0,
                  'replay_buffer_samples': 32,
                  'eps': EpsilonGreedy(eps_initial=0.5, decay=0.00001, eps_min=0.01, actions_pool=list(range(19))),
                  'replay_buffer': ContinuousBuffer(buffer_size=10000)}

        config.update(model_config)

        return config

    def _build_for_double_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn(dueling=False)
        config_dict.update({'name': os.path.join(self.folder, 'DoubleDQN'),
                            'double': True})

        return config_dict

    def _build_for_dueling_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn(dueling=True)
        config_dict.update({'name': os.path.join(self.folder, 'DuelingDQN'),
                            'double': False})

        return config_dict

    def _build_for_double_dueling_dqn(self) -> Dict[str, Any]:
        config_dict = self._build_for_dqn(dueling=True)
        config_dict.update({'name': os.path.join(self.folder, 'DoubleDuelingDQN'),
                            'double': True})

        return config_dict

    def _build_for_linear_q(self) -> Dict[str, Any]:
        if "-simple115v2-v0" not in self.env_spec:
            raise NotImplementedError("LinearQ only supports Simple115v2 wrapper.")

        return {"name": 'linear_q',
                "env_spec": self.env_spec,
                "eps": EpsilonGreedy(eps_initial=0.9, decay=0.001, eps_min=0.01, decay_schedule='linear',
                                     actions_pool=list(range(19)))}

    def _build_for_random(self) -> Dict[str, Any]:
        return {'name': os.path.join(self.folder, 'RandomAgent'),
                'env_spec': self.env_spec}
