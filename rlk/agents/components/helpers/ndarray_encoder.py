from json import JSONEncoder
from typing import Any, List

import numpy as np


class NDArrayEncoder(JSONEncoder):
    """Encoder for numpy arrays -> json"""
    def default(self, obj) -> List[Any]:
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return JSONEncoder.default(self, obj)
