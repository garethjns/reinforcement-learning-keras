from reinforcement_learning_keras.enviroments.doom.doom_default_config import DoomDefaultConfig


class VizDoomDeathMatchConfig(DoomDefaultConfig):
    """Defines specific config for this env."""
    env_spec = 'VizdoomDeathMatch-v0'
