from typing import Optional, List

import numpy as np

from rlk.environments.doom.bots.script_library import SCRIPTS


class ScriptedBot:
    """A bot that blindly follows a script of predefined actions."""
    _scripts = SCRIPTS
    script_n: int
    script: List[int]
    panic_action: int

    def __init__(self, env_name: str, script_n: Optional[int] = None, panic_action: int = 3):
        self.env_name = env_name

        self._step = 0
        self.using_script(script_n, panic_action)

    def using_script(self, script_n: Optional[int] = None, panic_action: int = 3) -> "ScriptedBot":
        if script_n is None:
            script_n = np.random.choice(list(self._scripts[self.env_name].keys()))

        self.panic_action = panic_action
        self.script_n = script_n
        self.script = self._scripts[self.env_name][self.script_n]

        return self

    def get_action(self) -> int:
        if self._step < len(self.script):
            action = self.script[self._step]
        else:
            action = self.panic_action

        self._step += 1

        return action

    def reset(self):
        self._step = 0
