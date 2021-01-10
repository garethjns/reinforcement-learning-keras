from typing import Dict, Optional, List, Any

import gym
import numpy as np

from rlk.environments.doom.vizdoom_corridor_config import VizDoomCorridorConfig  # noqa

VD_CORRIDOR_ACTION_MAP = {'a': 0, 'd': 1, ' ': 2, 'w': 3, 's': 4, 'q': 5, 'e': 6}


def get_user_action(action_map: Optional[Dict[str, int]] = None, action_space: Optional[List[int]] = None) -> int:
    raw_action = input("Enter action...")

    # See if this is a mappable action (eg. 'w')
    if action_map is not None:
        mapped_action = action_map.get(raw_action, raw_action)

    else:
        mapped_action = raw_action

    # Try and convert the action to int, should be possible if valid
    try:
        mapped_action = int(mapped_action)
    except ValueError:
        mapped_action = None

    # If the action is invalid, either sample from action space if supplied, or fallback to 0
    if (action_space is not None) and (mapped_action not in action_space):
        action = int(np.random.choice(action_space))
    elif mapped_action is None:
        action = 0
    else:
        # Action is valid.
        action = mapped_action

    if mapped_action is None:
        print(f"Invalid action {raw_action} ({mapped_action}), using {action} instead")
    else:
        print(f"Valid action {raw_action} ({mapped_action}), using {action}.")

    return action


def play_manual_game(env: gym.Env) -> Dict[str, List[Any]]:
    env.reset()
    env.render()

    done = False
    history = {'state': [], 'action': [], 'reward': [], 'done': []}
    while not done:
        action_ = get_user_action(action_map=VD_CORRIDOR_ACTION_MAP, action_space=list(range(env.action_space.n)))

        s, r, done, _ = env.step(action_)
        history['state'].append(s)
        history['action'].append(action_)
        history['reward'].append(r)
        history['done'].append(done)
        env.render()

    return history


if __name__ == "__main__":
    """
    7 available actions: 
    0) move left (a)
    1) move right (d)
    2) shoot (attack) (space)
    3) move forward (w)
    4) move backward (s)
    5) turn left (q)
    6) turn right (e)
    """

    env_ = gym.make('VizdoomCorridor-v0')

    history = play_manual_game(env_)

    print(history['action'])
