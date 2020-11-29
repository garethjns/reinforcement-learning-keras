import unittest

from tests.unit.environments.gfootball.environment_processing.fixtures.raw_obs_fixture import RawObsFixture

try:
    from rlk.environments.gfootball.environment_processing.raw_obs import RawObs

    GFOOTBALL_AVAILABLE = True
except ImportError:

    GFOOTBALL_AVAILABLE = False


    class RawObs:
        pass


@unittest.skipUnless(GFOOTBALL_AVAILABLE, "GFootball not available in this env.")
class TestRawObs(unittest.TestCase):

    def setUp(self):
        self._raw_obs_fixture = RawObsFixture()
        self.sut = RawObs(using=RawObs.standard_keys)
        self.sut.set_obs(self._raw_obs_fixture.data)

    def test_process_list_field(self):
        # Act
        obs = self.sut.process_key('ball')

        # Assert
        self.assertEqual(2, len(obs.shape))
        self.assertEqual(1, obs.shape[0])

    def test_process_non_list_field(self):
        # Act
        obs = self.sut.process_key('active')

        # Assert
        self.assertEqual(2, len(obs.shape))
        self.assertEqual(1, obs.shape[0])

    def test_process_non_flat_field(self):
        # Act
        obs = self.sut.process_key('left_team')

        # Assert
        self.assertEqual(2, len(obs.shape))
        self.assertEqual(1, obs.shape[0])

    def test_process(self):
        # Act
        raw_obs = self.sut.process()

        # Assert
        self.assertEqual(self.sut.shape, raw_obs.shape)

    def test__add_distance_to_ball(self):
        # Act
        distance_to_ball = self.sut._add_distance_to_ball()

        self.assertEqual((1, self.sut.distance_to_ball_n), distance_to_ball.shape)

    def test_with_obs_indexed_out_of_list(self):
        # Arrange
        ro = RawObs().set_obs(self._raw_obs_fixture.data[0])

        # Act
        raw_obs = ro.process()

        # Assert
        self.assertEqual(ro.shape, raw_obs.shape)

    def test_all_keys(self):
        for key in self.sut.standard_keys:
            obs = self.sut.process_key(key)
            self.assertEqual(getattr(self.sut, f"{key}_n"), obs.shape[1])

    def test_process_returns_none_if_nothing_to_do(self):
        # Arrange
        ro = RawObs(using=[]).set_obs(self._raw_obs_fixture.data[0])

        # Act
        raw_obs = ro.process()

        # Assert
        self.assertEqual((1, 0), ro.shape)
        self.assertIsNone(raw_obs)
