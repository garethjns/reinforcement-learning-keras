from reinforcement_learning_keras.enviroments.doom.doom_default_config import DoomDefaultConfig


class VizDoomTakeCoverConfig(DoomDefaultConfig):
    """Defines specific config for this env."""
    env_spec = 'VizdoomTakeCover-v0'
    n_actions = 2
