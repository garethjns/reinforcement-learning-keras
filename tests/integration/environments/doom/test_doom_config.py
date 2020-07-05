import unittest

from tests.integration.environments.atari.pong.test_pong_config import TestPongStackEnvironment, TestPongDiffEnvironment

try:
    from reinforcement_learning_keras.enviroments.doom.doom_config import DoomConfig

    run_tests = True
except ImportError:
    run_tests = False

    class DoomConfig:
        """Dummy as real failed import and it's used in test defs."""
        wrapped_env = None

        def __init__(self, *args, **kwargs):
            pass


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironment(TestPongStackEnvironment):
    _sut = DoomConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (128, 96, 3)
    _n_steps = 20
    _show = False


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnv(TestPongDiffEnvironment):
    _sut = DoomConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (128, 96, 1)
    _n_steps = 20
    _show = False


del TestPongStackEnvironment
del TestPongDiffEnvironment
