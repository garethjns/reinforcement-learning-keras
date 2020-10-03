from reinforcement_learning_keras.enviroments.cart_pole.cart_pole_config import CartPoleConfig
from tests.unit.environments.atari.pong.test_pong_config import TestPongConfig


class TestCartPoleConfig(TestPongConfig):
    def setUp(self):
        self._sut = CartPoleConfig
        self._agent_type = 'dueling_dqn'


del TestPongConfig
