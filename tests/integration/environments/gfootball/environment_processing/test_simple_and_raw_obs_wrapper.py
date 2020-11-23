import json
import os
import tempfile
import unittest

import gym
import numpy as np

from rlk.environments.gfootball.environment_processing.raw_obs import RawObs
from tests.unit.environments.gfootball.environment_processing.fixtures.raw_obs_fixture import RawObsFixture

try:
    from gfootball.env.config import Config
    from gfootball.env.football_env import FootballEnv
    from rlk.environments.gfootball.environment_processing.simple_and_raw_obs_wrapper import SimpleAndRawObsWrapper
    from rlk.environments.gfootball.register_environments import register_all

    GFOOTBALL_AVAILABLE = True
except ImportError:
    GFOOTBALL_AVAILABLE = False


@unittest.skipUnless(GFOOTBALL_AVAILABLE, "GFootball not available in this env.")
class TestSimpleAndRawObsWrapper(unittest.TestCase):
    _raw_obs_fixture = RawObsFixture()

    @classmethod
    def setUpClass(cls) -> None:
        register_all()
        cls.tmp_dir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp_dir.cleanup()

    def setUp(self) -> None:
        self._env = FootballEnv(config=Config())
        self._sut = SimpleAndRawObsWrapper

    def test_shapes_as_expected_with_env_on_reset(self):
        # Arrange
        wrapped_env = self._sut(self._env)

        # Act
        obs = wrapped_env.reset()

        # Assert
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(wrapped_env.observation_space.shape, obs.shape)
        self.assertEqual((RawObs().shape[1] + 115,), obs.shape)

    def test_shapes_as_expected_with_kaggle_env(self):
        # Arrange
        wrapped_env = self._sut(gym.make("GFootball-kaggle_11_vs_11-v0"))

        # Act
        obs = wrapped_env.reset()

        # Assert
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(wrapped_env.observation_space.shape, obs.shape)
        self.assertEqual((RawObs().shape[1] + 115,), obs.shape)

    def test_shapes_as_expected_without_env_on_reset(self):
        # Arrange
        raw_obs = self._env.reset()

        # Act
        processed_obs = self._sut.process_obs(raw_obs)

        # Assert
        self.assertIsInstance(processed_obs, np.ndarray)
        self.assertEqual((RawObs().shape[1] + 115,), processed_obs.shape)

    def test_shapes_as_expected_with_env_on_reset_with_all(self):
        # Arrange
        wrapped_env = self._sut(self._env, raw_using=RawObs.standard_keys)

        # Act
        obs = wrapped_env.reset()

        # Assert
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(wrapped_env.observation_space.shape, obs.shape)
        self.assertEqual((RawObs(using=RawObs.standard_keys).shape[1] + 115,), obs.shape)

    def test_shapes_as_expected_with_env_on_step(self):
        # Arrange
        wrapped_env = self._sut(self._env)
        wrapped_env.reset()

        # Act
        obs, reward, done, info = wrapped_env.step(0)

        # Assert
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(wrapped_env.observation_space.shape, obs.shape)
        self.assertEqual((RawObs().shape[1] + 115,), obs.shape)

    def test_shapes_as_expected_without_env_on_step(self):
        # Arrange
        _ = self._env.reset()
        obs, reward, done, info = self._env.step(0)

        # Act
        obs = self._sut.process_obs(obs)

        # Assert
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual((RawObs().shape[1] + 115,), obs.shape)

    def test_dump_created_when_path_set(self):
        # Arrange
        raw_dump_path = f"{self.tmp_dir.name}/1.json"
        wrapped_env = self._sut(self._env, raw_dump_path=raw_dump_path)

        # Act
        _ = wrapped_env.reset()
        obs, _, _, _ = wrapped_env.step(0)

        # Assert
        self.assertTrue(os.path.exists(raw_dump_path))
        with open(raw_dump_path, 'r') as f:
            loaded_json = json.load(f)
        self.assertListEqual(['players_raw'], list(loaded_json.keys()))
        self.assertListEqual(sorted(RawObs.standard_keys), sorted(list(loaded_json['players_raw'][0].keys())))

    def test_dumps_but_does_not_add_to_returned_obs_when_using_is_empty(self):
        # Arrange
        raw_dump_path = f"{self.tmp_dir.name}/2.json"
        wrapped_env = self._sut(self._env, raw_dump_path=raw_dump_path, raw_using=[])

        # Act
        _ = wrapped_env.reset()
        obs, _, _, _ = wrapped_env.step(0)

        # Assert
        self.assertEqual((115,), obs.shape)
        self.assertTrue(os.path.exists(raw_dump_path))
        with open(raw_dump_path, 'r') as f:
            loaded_json = json.load(f)
        self.assertListEqual(['players_raw'], list(loaded_json.keys()))
        self.assertListEqual(sorted(RawObs.standard_keys), sorted(list(loaded_json['players_raw'][0].keys())))

