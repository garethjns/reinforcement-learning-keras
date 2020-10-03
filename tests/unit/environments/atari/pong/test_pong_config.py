import unittest

from reinforcement_learning_keras.enviroments.atari.pong.pong_config import PongConfig


class TestPongConfig(unittest.TestCase):
    def setUp(self):
        self._sut = PongConfig
        self._agent_type = 'dueling_dqn'

    def test_fails_without_agent_type(self):
        self.assertRaises(TypeError, lambda: self._sut())

    def test_specify_agent_type_as_kwarg(self):
        # Act
        config = self._sut(agent_type=self._agent_type)

        # Assert
        self.assertEqual(self._agent_type, config.agent_type)

    def test_specify_agent_type_as_arg(self):
        # Act
        config = self._sut(self._agent_type)

        # Assert
        self.assertEqual(self._agent_type, config.agent_type)
