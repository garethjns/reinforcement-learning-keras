import os
from typing import Any, Dict

from reinforcement_learning_keras.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from reinforcement_learning_keras.agents.models.dense_nn import DenseNN
from reinforcement_learning_keras.agents.models.splitter_conv_nn import SplitterConvNN
from reinforcement_learning_keras.agents.q_learning.exploration.epsilon_greedy import EpsilonGreedy
from reinforcement_learning_keras.environments.config_base import ConfigBase
from reinforcement_learning_keras.environments.gfootball.environment_processing.simple_and_smm_obs_wrapper import \
    SimpleAndSMMObsWrapper
from reinforcement_learning_keras.environments.gfootball.environment_processing.smm_frame_process_wrapper import \
    SMMFrameProcessWrapper


class GFootballConfig(ConfigBase):
    """Defines config for cart_pole."""
    supported_agents = ('linear_q', 'dqn', 'double_dqn', 'dueling_dqn', 'double_dueling_dqn')
    gpu_memory: int = 2048

    def __init__(self, *args, env_spec: str = "GFootball-11_vs_11_kaggle-simple115v2-v0", **kwargs):
        self.env_spec = env_spec
        super().__init__(*args, **kwargs)

    @property
    def _default_training_history_kwargs(self) -> Dict[str, Any]:
        return {"plotting_on": self.plot_during_training,
                "plot_every": 25, "rolling_average": 12}

    @staticmethod
    def _build_with_dense_model(dueling: bool = False) -> Dict[str, Any]:
        return {"model_architecture": DenseNN(observation_shape=(115,), n_actions=19, dueling=dueling)}

    @staticmethod
    def _build_with_splitter_conv_model(dueling: bool = False) -> Dict[str, Any]:
        return {"model_architecture": SplitterConvNN(observation_shape=(72, 96, 4), n_actions=19, dueling=dueling),
                "env_wrappers": [SMMFrameProcessWrapper]}

    @staticmethod
    def _build_with_splitter_conv_and_dense_model(dueling: bool = False) -> Dict[str, Any]:
        return {"model_architecture": SplitterConvNN(observation_shape=(72, 96, 4),
                                                     additional_dense_input_shape=(115,),
                                                     n_actions=19, dueling=dueling),
                "env_wrappers": [SimpleAndSMMObsWrapper, SMMFrameProcessWrapper]}

    def _build_for_dqn(self, dueling: bool=False) -> Dict[str, Any]:
        if "-simple115v2-v0" in self.env_spec:
            model_config = self._build_with_dense_model(dueling)
        if "-SMM-v0" in self.env_spec:
            model_config = self._build_with_splitter_conv_model(dueling)
        if "-SimpleSMM-" in self.env_spec:
            model_config = self._build_with_splitter_conv_and_dense_model(dueling)

        config = {'name': os.path.join(self.folder, 'DeepQAgent'),
                  'env_spec': self.env_spec,
                  'gamma': 0.99,
                  'final_reward': 0,
                  'replay_buffer_samples': 75,
                  'eps': EpsilonGreedy(eps_initial=1.6, decay=0.00001, eps_min=0.01),
                  'replay_buffer': ContinuousBuffer(buffer_size=50000)}

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
                "eps": EpsilonGreedy(eps_initial=0.9, decay=0.001, eps_min=0.01, decay_schedule='linear')}

    def _build_for_random(self) -> Dict[str, Any]:
        return {'name': os.path.join(self.folder, 'RandomAgent'),
                'env_spec': self.env_spec}
