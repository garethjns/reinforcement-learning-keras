import unittest

from agents.agent_base import AgentBase
from agents.cart_pole.policy_gradient.reinforce_agent import ReinforceAgent
from agents.cart_pole.q_learning.deep_q_agent import DeepQAgent
from agents.cart_pole.q_learning.dueling_deep_q_agent import DuelingDeepQAgent
from agents.cart_pole.q_learning.linear_q_agent import LinearQAgent
from agents.cart_pole.random.random_agent import RandomAgent
from experiment.agent_experiment import AgentExperiment


class TestAgentExperiment(unittest.TestCase):
    _sut = AgentExperiment

    def _run_exp(self, agent_class: AgentBase, n_jobs: int = 1):
        # Arrange
        exp = AgentExperiment(env_spec="CartPole-v0",
                              agent_class=agent_class,
                              n_reps=3,
                              n_jobs=n_jobs,
                              n_episodes=4,
                              max_episode_steps=4)

        # Act
        exp.run()

        # Assert
        self.assertEqual(3, len(exp._trained_agents))
        self.assertEqual(3, len(exp.agent_scores))
        for a in exp._trained_agents:
            self.assertEqual(4, len(a.history.history))

    def test_short_random_agent_run_completes_with_expected_outputs(self):
        self._run_exp(RandomAgent)

    def test_short_random_agent_parallel_run_completes_with_expected_outputs(self):
        self._run_exp(RandomAgent, n_jobs=3)

    def test_short_linear_q_agent_run_completes_with_expected_outputs(self):
        self._run_exp(LinearQAgent)

    def test_short_linear_q_agent_parallel_run_completes_with_expected_outputs(self):
        self._run_exp(LinearQAgent, n_jobs=3)

    def test_short_deep_q_agent_run_completes_with_expected_outputs(self):
        DeepQAgent.set_tf(256)
        self._run_exp(DeepQAgent)

    def test_short_deep_q_agent_parallel_run_completes_with_expected_outputs(self):
        gpu = DeepQAgent.set_tf(256)
        if not gpu:
            self._run_exp(DeepQAgent)

    def test_short_dueling_deep_q_agent_run_completes_with_expected_outputs(self):
        DuelingDeepQAgent.set_tf(256)
        self._run_exp(DuelingDeepQAgent)

    def test_short_dueling_deep_q_agent_parallel_run_completes_with_expected_outputs(self):
        gpu = DuelingDeepQAgent.set_tf(256)
        if not gpu:
            self._run_exp(DuelingDeepQAgent)

    def test_short_reinforce_agent_run_completes_with_expected_outputs(self):
        ReinforceAgent.set_tf(256)
        self._run_exp(ReinforceAgent)

    def test_short_reinforce_agent_parallel_run_completes_with_expected_outputs(self):
        gpu = ReinforceAgent.set_tf(256)
        if not gpu:
            self._run_exp(ReinforceAgent)
