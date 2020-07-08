import unittest
from unittest.mock import MagicMock

import reinforcement_learning_keras.enviroments.doom as doom
from tests.integration.environments.atari.pong.test_pong_config import TestPongStackEnvironment, TestPongDiffEnvironment

if doom.AVAILABLE:
    run_tests = True
else:
    run_tests = False
    doom = MagicMock()


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = doom.VizDoomBasicConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = doom.VizDoomCorridorConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = doom.VizDoomDeathmatchConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = doom.VizDoomDefendCenterConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = doom.VizDoomDefendLineConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = doom.VizDoomHealthGatheringConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = doom.VizDoomHealthGatheringSupremeConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = doom.VizDoomMyWayHomeConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = doom.VizDoomPredictPositionConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (90, 160, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = doom.VizDoomTakeCoverConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)
    _show = False


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongDiffEnvironment):
    _sut = doom.VizDoomBasicConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = doom.VizDoomCorridorConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = doom.VizDoomDeathmatchConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = doom.VizDoomDefendCenterConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = doom.VizDoomDefendLineConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = doom.VizDoomHealthGatheringConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = doom.VizDoomHealthGatheringSupremeConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = doom.VizDoomMyWayHomeConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = doom.VizDoomPredictPositionConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (90, 160, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = doom.VizDoomTakeCoverConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)
    _show = False


del TestPongStackEnvironment
del TestPongDiffEnvironment
