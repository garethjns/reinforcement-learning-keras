import uuid
import warnings
from collections import Callable
from typing import Any, Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed

from rlk.agents.components.helpers.virtual_gpu import VirtualGPU
from rlk.agents.components.history.training_history import TrainingHistory
from rlk.agents.components.replay_buffers.continuous_buffer import ContinuousBuffer
from rlk.agents.q_learning.deep_q_agent import DeepQAgent
from rlk.environments.config_base import ConfigBase
from rlk.environments.gfootball.gfootball_config import GFootballConfig


class MultiTrainer:
    """Train on multiple environments."""
    agent_class: Callable
    agent_config: Dict[str, Any]
    agent: DeepQAgent
    gpu_memory_limit: int
    device_id: int = 0

    def set_agent(self, agent_class: Callable, agent_config: ConfigBase, agent_kwargs: Dict[str, any],
                  gpu_memory_limit: int = 512):
        self.gpu_memory_limit = gpu_memory_limit
        VirtualGPU(gpu_device_id=self.device_id, gpu_memory_limit=self.gpu_memory_limit)

        self.agent_class = agent_class
        config = agent_config.build()
        config.update(agent_kwargs)
        self.agent_config = config
        self.agent = agent_class(**self.agent_config)

    def train(self, n_group_training_steps: int = 2, **kwargs):
        for _ in range(n_group_training_steps):
            self.train_group(**kwargs)

    def train_group(self, n_jobs: int = 1, n_rounds: int = 3, max_episode_steps: int = 10, **kwargs) -> None:

        self.agent_config['eps'] = self.agent.eps
        self.agent_config['replay_buffer'] = self.agent.replay_buffer

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_agent)(agent_class=self.agent_class, agent_config=self.agent_config,
                                     training_kwargs=kwargs, real_device_id=0, gpu_memory_limit=self.gpu_memory_limit)
            for _ in range(n_rounds))

        new_weights = [r[0] for r in results]
        replay_buffers = [r[1] for r in results]
        training_histories = [r[2] for r in results]

        self.update_main_weights(new_weights)
        self.update_main_epsilon(n_steps=max_episode_steps)
        self.update_main_replay_buffer(replay_buffers)
        self.update_main_training_history(training_histories)

    def update_main_training_history(self, training_histories: List[TrainingHistory]):

        for th in training_histories:
            self.agent.training_history.extend(th.history)

        self.agent.training_history.plot(metrics=["frames" ,"total_reward"], show=True)

    def update_main_replay_buffer(self, replay_buffers: List[ContinuousBuffer]):
        main_n = self.agent.replay_buffer.buffer_size
        # TODO: Should be shuffled
        for rpb in replay_buffers:
            update_n = min(rpb.n, int(main_n / len(replay_buffers)))
            samples = rpb.sample_batch(update_n)
            for s_idx in range(update_n):
                self.agent.replay_buffer.append((samples[0][s_idx], samples[1][s_idx],
                                                 samples[2][s_idx], samples[3][s_idx]))

    def update_main_epsilon(self, n_steps):
        for _ in range(n_steps):
            self.agent.eps.eps_current = self.agent.eps._decay()

    def update_main_weights(self, new_weights: List[np.ndarray]) -> None:
        current_weights = self.agent.get_weights()

        # Iterate over new weight sets and get diff to current
        diff_weights = []
        for nw in new_weights:
            diff_weights.append([nwl - cwl for nwl, cwl in zip(nw, current_weights)])

        # Iterate within diff weights and mean each group, calc diff to current group
        n_weights = len(diff_weights[0])
        weights_update = []
        for wi in range(n_weights):
            layer_diff_mean = np.mean([dw[wi] for dw in diff_weights], axis=0)
            weights_update.append(current_weights[wi] + layer_diff_mean)

        self.agent.set_weights(current_weights)

    @staticmethod
    def _fit_agent(agent_class: Callable, agent_config: Dict[str, Any],
                   training_kwargs: Dict[str, Any], real_device_id: int = 0,
                   gpu_memory_limit: int = 512) -> Tuple[np.ndarray, ContinuousBuffer, TrainingHistory]:
        VirtualGPU(gpu_device_id=real_device_id, gpu_memory_limit=gpu_memory_limit)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)

            agent_config["name"] = f"{agent_config.get('name', 'Agent')}_{str(uuid.uuid1())}"

            agent: DeepQAgent = agent_class(**agent_config)
            agent.train(**training_kwargs)

        return agent.get_weights(), agent.replay_buffer, agent.training_history


if __name__ == "__main__":
    mt = MultiTrainer()
    mt.set_agent(agent_class=DeepQAgent,
                 agent_config=GFootballConfig('double_dueling_dqn', env_spec="GFootball-academy_empty_goal_close-v0"),
                 agent_kwargs={'env_builder_kwargs': {'remote': True, 'ip': '192.168.68.124'}})

    # mt.train_group(n_jobs=3, n_rounds=3, n_episodes=3, max_episode_steps=10)
    mt.train(n_group_training_steps=30, n_jobs=3, n_rounds=3, n_episodes=10, max_episode_steps=3000)
