from typing import Any, Callable

from tqdm import tqdm


class TQDMHandler:
    """
    Turns TQDM waitbars on or off depending on a verbosity setting.

    Useful to avoid having to change the for loop manually to switch.
    """

    def __init__(self):
        self.tqdm_runner: Callable = tqdm

    @staticmethod
    def _fake_tqdm(x: Any) -> Any:
        return x

    def set_tqdm(self, verbose: bool = False) -> None:
        """Turn tqdm on or of depending on verbosity setting."""
        if verbose:
            self.tqdm_runner = tqdm
        else:
            self.tqdm_runner = self._fake_tqdm
