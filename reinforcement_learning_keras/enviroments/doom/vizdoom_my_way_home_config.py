from reinforcement_learning_keras.enviroments.doom.doom_default_config import DoomDefaultConfig


class VizDoomMyWayHomeConfig(DoomDefaultConfig):
    """Defines specific config for this env."""
    env_spec = 'VizdoomMyWayHome-v0'
    n_actions = 3
    res_scale = 0.2