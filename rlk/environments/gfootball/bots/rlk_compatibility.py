import json
from collections import Callable
from functools import wraps

import numpy as np

from rlk.environments.gfootball.bots.bot_config import BotConfig


def rlk_compatibility(agent: Callable) -> Callable:
    """
    Decorator to get obs from dump on disk if passed obs is None.

    Compatible with both naked agent function and agent function already decorated with @human_readable_agent. In the
    latter case, the decorator requires the obs to not be None, so this should be the outer decorator. Eg:

    @rlk_compatibility
    @human_readable_agent
    def agent(obs):
        pass

    Path to json dump to load from is set in the BotConfig(). This should work fine with relative paths.
    """

    # @wraps is required to make obs_getter pickleable (otherwise local get_obs can't be pickled)
    @wraps(agent)
    def get_obs(obs):
        rlk_compat = False

        if (obs is None) or isinstance(obs, np.ndarray):
            # Either agent is being passed no obs, or it's being passed a processed observation during rl training.
            # In both cases, discard and replace with last dumped raw observation by SimpleAndRawObsWrapper.
            rlk_compat = True
            with open(BotConfig().json_dump_path, 'r') as f:
                obs = json.load(f)

        if rlk_compat:
            # If rlk is running this, we just want the action
            action = agent(obs)[0]
        else:
            # Where as Kaggle wants a list of actions for each (1) agent
            action = agent(obs)

        return action

    return get_obs
