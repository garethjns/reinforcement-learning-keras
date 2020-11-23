import unittest

import gym

from rlk.environments.gfootball.environment_processing.raw_obs import RawObs

try:
    from gfootball.env.config import Config
    from gfootball.env.football_env import FootballEnv
    from rlk.environments.gfootball.register_environments import register_all

    GFOOTBALL_AVAILABLE = True
except ImportError:
    GFOOTBALL_AVAILABLE = False


@unittest.skipUnless(GFOOTBALL_AVAILABLE, "GFootball not available in this env.")
class TestRawObs(unittest.TestCase):
    """
    Note that this test tests with environments, rather than the fixture. The fixture matches the output passed to
    agent when evaluated with env.make() and env.run() using kaggle_environments.make. This differs slightly to the raw
    observations returned by the envs (gym.make or gfootball.make), which contains arrays in place of lists for
    some fields.

    The processing in Raw obs must work with both, so can't do things like [] + [] as if one of those is an array,
    something different happens....
    """

    @classmethod
    def setUpClass(cls) -> None:
        register_all()

    def setUp(self):
        self._kaggle_env = gym.make("GFootball-kaggle_11_vs_11-v0")
        self._gfootball_env = self._env = FootballEnv(config=Config())
        self._sut = RawObs()

    def test_process_active_output_shapes_as_expected_with_kaggle_env(self):
        # Arrange
        ro = self._sut.set_obs(self._kaggle_env.reset())

        # Act
        active = ro.process_key('active')

        # Assert
        self.assertEqual((1, RawObs.active_n), active.shape)

    def test_process_active_output_shapes_as_expected_with_gf_env(self):
        # Arrange
        ro = self._sut.set_obs(self._gfootball_env.reset())

        # Act
        active = ro.process_key('active')

        # Assert
        self.assertEqual((1, RawObs.active_n), active.shape)

    def test_process_ball_output_shapes_as_expected_with_kaggle_env(self):
        # Arrange
        ro = self._sut.set_obs(self._kaggle_env.reset())

        # Act
        active = ro.process_key('ball')

        # Assert
        self.assertEqual((1, RawObs.ball_n), active.shape)

    def test_process_ball_info_output_shapes_as_expected_with_gf_env(self):
        # Arrange
        ro = self._sut.set_obs(self._gfootball_env.reset())

        # Act
        active = ro.process_key('ball')

        # Assert
        self.assertEqual((1, RawObs.ball_n), active.shape)

    def test_process_tired_factor_output_shapes_as_expected_with_kaggle_env(self):
        # Arrange
        ro = self._sut.set_obs(self._kaggle_env.reset())

        # Act
        active = ro.process_key('left_team_tired_factor')

        # Assert
        self.assertEqual((1, RawObs.left_team_tired_factor_n), active.shape)

    def test_process_tired_factor_output_shapes_as_expected_with_gf_env(self):
        # Arrange
        ro = self._sut.set_obs(self._gfootball_env.reset())

        # Act
        active = ro.process_key('right_team_tired_factor')

        # Assert
        self.assertEqual((1, RawObs.right_team_tired_factor_n), active.shape)

    def test_output_shape_as_expected_with_kaggle_env_reset(self):
        # Arrange
        obs = self._kaggle_env.reset()

        # Act
        raw_obs = RawObs.convert_observation(obs)

        # Assert
        self.assertEqual(RawObs().shape, raw_obs.shape)

    def test_output_shape_as_expected_with_gfootball_env_reset(self):
        obs = self._gfootball_env.reset()
        # Act
        raw_obs = RawObs.convert_observation(obs)

        # Assert
        self.assertEqual(RawObs().shape, raw_obs.shape)

    def test_process_all_shape_as_expected_with_kaggle_env(self):
        # Arrange
        raw_obs = self._kaggle_env.reset()

        # Act
        processed_obs = self._sut.set_obs(raw_obs).process()

        self.assertEqual(RawObs().shape, processed_obs.shape)

    def test_process_defaults_shape_as_expected_with_kaggle_env(self):
        # Arrange
        raw_obs = self._kaggle_env.reset()

        # Act
        processed_obs = RawObs(using=RawObs.standard_keys).set_obs(raw_obs).process()

        self.assertEqual(RawObs(using=RawObs.standard_keys).shape, processed_obs.shape)

    def test_process_all_shape_as_expected_with_gfootball_env(self):
        # Arrange
        raw_obs = self._gfootball_env.reset()

        # Act
        processed_obs = self._sut.set_obs(raw_obs).process()

        self.assertEqual(RawObs().shape, processed_obs.shape)

    def test_process_defaults_shape_as_expected_with_gfootball_env(self):
        # Arrange
        raw_obs = self._gfootball_env.reset()

        # Act
        processed_obs = RawObs(using=RawObs.standard_keys).set_obs(raw_obs).process()

        self.assertEqual(RawObs(using=RawObs.standard_keys).shape, processed_obs.shape)
