import os
import pathlib


class BotConfig:
    json_dump_path = '/home/gareth/json_dump_path2/dump.json'

    def __init__(self):
        pathlib.Path(os.path.split(self.json_dump_path)[0]).mkdir(exist_ok=True)
