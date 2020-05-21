import unittest

from agents.cart_pole.q_learning.linear_q_agent import LinearQLearningAgent
from experiment.agent_experiment import AgentExperiment


class TestAgentExperiment(unittest.TestCase):
    _sut = AgentExperiment

    def test_short_linear_q_run_completes_with_expected_outputs(self):
        # Arrange
        exp = AgentExperiment(env_spec="CartPole-v0",
                              agent_class=LinearQLearningAgent,
                              n_reps=3,
                              n_jobs=3,
                              n_episodes=4,
                              max_episode_steps=4)

        # Act
        exp.run()

        # Assert
        self.assertEqual(3, len(exp._trained_agents))
        self.assertEqual(3, len(exp.agent_scores))
