from agents.cart_pole.random.random_agent import RandomAgent
from tests.integration.agents.cart_pole.random.test_random_agent import TestRandomAgent


class TestReinforceAgent(TestRandomAgent):
    _sut = RandomAgent
    _fn = 'test_reinforce_save.agents'
