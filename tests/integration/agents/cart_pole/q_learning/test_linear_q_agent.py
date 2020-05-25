from agents.cart_pole.q_learning.linear_q_agent import LinearQAgent
from tests.integration.agents.cart_pole.random.test_random_agent import TestRandomAgent


class TestDeepQLearningAgent(TestRandomAgent):
    _sut = LinearQAgent
    _fn = 'test_linear_save.agent'
