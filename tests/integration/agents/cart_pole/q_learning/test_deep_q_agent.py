from agents.cart_pole.q_learning.deep_q_agent import DeepQAgent

from tests.integration.agents.cart_pole.random.test_random_agent import TestRandomAgent


class TestDeepQLearningAgent(TestRandomAgent):
    _sut = DeepQAgent
    _fn = 'test_dqn_save.agent'

    @classmethod
    def setUp(cls):
        cls._sut.set_tf(256)
