import json
import os
import unittest

import numpy as np

from tests.unit.environments.gfootball.environment_processing.fixtures.raw_obs_fixture import RawObsFixture

try:
    from rlk.environments.gfootball.bots.open_rules_bot import agent
    from rlk.environments.gfootball.bots.bot_config import BotConfig

    KAGGLE_ENVS_AVAILABLE = True
except ImportError:
    KAGGLE_ENVS_AVAILABLE = False


@unittest.skipUnless(KAGGLE_ENVS_AVAILABLE, "Kaggle envs not available, GFootball probably not installed")
class TestRulesBot(unittest.TestCase):
    _raw_obs_fixture = RawObsFixture()

    def setUp(self) -> None:
        self._bot_config = BotConfig()
        with open(self._bot_config.json_dump_path, 'w') as f:
            json.dump({'players_raw': self._raw_obs_fixture.data}, f)

    def tearDown(self) -> None:
        try:
            os.remove(self._bot_config.json_dump_path)
            os.rmdir(os.path.split(self._bot_config.json_dump_path)[0])
        except FileNotFoundError:
            pass

    def test_bot_returns_action_when_passed_raw_obs(self):
        # Act
        action = agent({'players_raw': self._raw_obs_fixture.data})

        # Assert
        self.assertIsInstance(action, list)
        self.assertIsInstance(action[0], int)
        self.assertIn(action[0], list(range(19)))

    def test_bot_returns_action_when_not_passed_obs(self):
        # Act
        action = agent(obs=None)

        # Assert
        self.assertIsInstance(action, int)
        self.assertIn(action, list(range(19)))

    def test_bot_returns_action_when_passed_array_obs(self):
        # Act
        action = agent(obs=np.array([]))

        # Assert
        self.assertIsInstance(action, int)
        self.assertIn(action, list(range(19)))
