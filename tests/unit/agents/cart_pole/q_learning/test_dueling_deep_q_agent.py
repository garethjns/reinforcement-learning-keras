from agents.cart_pole.q_learning.dueling_deep_q_agent import DuelingDeepQAgent
from tests.unit.agents.cart_pole.q_learning.test_deep_q_agent import TestDeepQAgent


class TestDuelingDeepQAgent(TestDeepQAgent):
    _sut = DuelingDeepQAgent
