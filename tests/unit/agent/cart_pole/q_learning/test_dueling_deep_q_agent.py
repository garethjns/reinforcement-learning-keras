from agents.cart_pole.q_learning.dueling_deep_q_agent import DuelingDeepQAgent
from tests.unit.agent.cart_pole.q_learning.test_deep_q_agent import TestDeepQAgent


class TestDeepQAgent(TestDeepQAgent):
    _sut = DuelingDeepQAgent
