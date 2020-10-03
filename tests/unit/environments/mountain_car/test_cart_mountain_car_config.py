from reinforcement_learning_keras.enviroments.mountain_car.mountain_car_config import MountainCarConfig
from tests.unit.environments.atari.pong.test_pong_config import TestPongConfig


class TestMountainCarConfig(TestPongConfig):
    def setUp(self):
        self._sut = MountainCarConfig
        self._agent_type = 'dueling_dqn'


del TestPongConfig
