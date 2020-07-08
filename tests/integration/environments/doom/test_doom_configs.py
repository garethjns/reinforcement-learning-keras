import unittest

from tests.integration.environments.atari.pong.test_pong_config import TestPongStackEnvironment, TestPongDiffEnvironment

try:
    from reinforcement_learning_keras.enviroments.doom.vizdoom_basic_config import VizDoomBasicConfig
    from reinforcement_learning_keras.enviroments.doom.vizdoom_corridor_config import VizDoomCorridorConfig
    from reinforcement_learning_keras.enviroments.doom.vizdoom_deathmatch_config import VizDoomDeathmatchConfig
    from reinforcement_learning_keras.enviroments.doom.vizdoom_defend_center_config import VizDoomDefendCenterConfig
    from reinforcement_learning_keras.enviroments.doom.vizdoom_defend_line_config import VizDoomDefendLineConfig
    from reinforcement_learning_keras.enviroments.doom.vizdoom_health_gathering_config import \
        VizDoomHealthGatheringConfig
    from reinforcement_learning_keras.enviroments.doom.vizdoom_health_gathering_supreme import \
        VizDoomHealthGatheringSupremeConfig
    from reinforcement_learning_keras.enviroments.doom.vizdoom_my_way_home_config import VizDoomMyWayHomeConfig
    from reinforcement_learning_keras.enviroments.doom.vizdoom_predict_position_config import \
        VizDoomPredictPositionConfig
    from reinforcement_learning_keras.enviroments.doom.vizdoom_take_cover_config import VizDoomTakeCoverConfig

    run_tests = True
except ImportError:
    run_tests = False


    class DoomConfig:
        wrapped_env = None

        def __init__(self, *args, **kwargs):
            """Dummy as real failed import and it's used in test defs."""
            pass


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = VizDoomBasicConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = VizDoomCorridorConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = VizDoomDeathmatchConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = VizDoomDefendCenterConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = VizDoomDefendLineConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = VizDoomHealthGatheringConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = VizDoomHealthGatheringSupremeConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = VizDoomMyWayHomeConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = VizDoomPredictPositionConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (90, 160, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongStackEnvironment):
    _sut = VizDoomTakeCoverConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)
    _show = False


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomStackEnvironments(TestPongDiffEnvironment):
    _sut = VizDoomBasicConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = VizDoomCorridorConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = VizDoomDeathmatchConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = VizDoomDefendCenterConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = VizDoomDefendLineConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = VizDoomHealthGatheringConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = VizDoomHealthGatheringSupremeConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = VizDoomMyWayHomeConfig(mode='diff', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 1)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = VizDoomPredictPositionConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (90, 160, 3)


@unittest.skipUnless(run_tests, reason='ViZDoomGym not installed')
class TestDoomDiffEnvironments(TestPongDiffEnvironment):
    _sut = VizDoomTakeCoverConfig(mode='stack', agent_type='dqn').wrapped_env
    _expected_shape = (96, 128, 3)
    _show = False


del TestPongStackEnvironment
del TestPongDiffEnvironment
