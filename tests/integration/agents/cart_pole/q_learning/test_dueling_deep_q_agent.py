from agents.cart_pole.q_learning.dueling_deep_q_agent import DuelingDeepQAgent

from tests.integration.agents.cart_pole.random.test_random_agent import TestRandomAgent


class TestDeepQLearningAgent(TestRandomAgent):
    _sut = DuelingDeepQAgent
    _fn = 'test_ddqn_save.agent'

    @classmethod
    def setUp(cls):
        cls._sut.set_tf(256)
