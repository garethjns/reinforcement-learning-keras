from reinforcement_learning_keras.enviroments.doom.doom_default_config import DoomDefaultConfig


class VizDoomCorridorConfig(DoomDefaultConfig):
    """Defines specific config for this env."""
    env_spec = 'VizdoomCorridor-v0'
    n_actions = 7
