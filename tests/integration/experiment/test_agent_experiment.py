import unittest

from agents.agent_base import AgentBase
from agents.components.helpers.virtual_gpu import VirtualGPU
from agents.policy_gradient.reinforce_agent import ReinforceAgent
from agents.q_learning.deep_q_agent import DeepQAgent
from agents.q_learning.linear_q_agent import LinearQAgent
from agents.random.random_agent import RandomAgent
from enviroments.cart_pole.cart_pole_config import CartPoleConfig
from experiment.agent_experiment import AgentExperiment


class TestAgentExperiment(unittest.TestCase):
    _sut = AgentExperiment
    _agent_config = CartPoleConfig

    def _run_exp(self, agent_class: AgentBase, agent_type: str, n_jobs: int = 1):
        # Arrange
        exp = AgentExperiment(agent_class=agent_class,
                              agent_config=self._agent_config(agent_type=agent_type),
                              n_reps=3,
                              n_jobs=n_jobs,
                              training_options={"n_episodes": 4,
                                                "max_episode_steps": 4})

        # Act
        exp.run()

        # Assert
        self.assertEqual(3, len(exp._trained_agents))
        self.assertEqual(3, len(exp.agent_scores))
        for a in exp._trained_agents:
            self.assertEqual(4, len(a.training_history.history))

    def test_short_random_agent_run_completes_with_expected_outputs(self):
        self._run_exp(RandomAgent, agent_type='random')

    def test_short_random_agent_parallel_run_completes_with_expected_outputs(self):
        self._run_exp(RandomAgent, agent_type='random', n_jobs=3)

    def test_short_linear_q_agent_run_completes_with_expected_outputs(self):
        self._run_exp(LinearQAgent, agent_type='linear_q')

    def test_short_linear_q_agent_parallel_run_completes_with_expected_outputs(self):
        self._run_exp(LinearQAgent, agent_type='linear_q', n_jobs=3)

    def test_short_deep_q_agent_run_completes_with_expected_outputs(self):
        VirtualGPU(256)
        self._run_exp(DeepQAgent, agent_type='dqn')

    def test_short_deep_q_agent_parallel_run_completes_with_expected_outputs(self):
        gpu = VirtualGPU(256)
        if not gpu:
            self._run_exp(DeepQAgent, agent_type='dqn')

    def test_short_reinforce_agent_run_completes_with_expected_outputs(self):
        VirtualGPU(256)
        self._run_exp(ReinforceAgent, agent_type='reinforce')

    def test_short_reinforce_agent_parallel_run_completes_with_expected_outputs(self):
        gpu = VirtualGPU(256)
        if not gpu:
            self._run_exp(ReinforceAgent, agent_type='reinforce')
