from reinforcement_learning_keras.enviroments.atari.space_invaders.space_invaders_config import SpaceInvadersConfig
from tests.integration.environments.atari.pong.test_pong_config import TestPongStackEnvironment, TestPongDiffEnvironment


class TestSpaceInvadersStackEnvironment(TestPongStackEnvironment):
    _sut = SpaceInvadersConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (84, 84, 3)
    _n_steps = 20
    _show = False


class TestSpaceInvadersDiffEnv(TestPongDiffEnvironment):
    _sut = SpaceInvadersConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (84, 84, 1)
    _n_steps = 20
    _show = False


del TestPongStackEnvironment
del TestPongDiffEnvironment
