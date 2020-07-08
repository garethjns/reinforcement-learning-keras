from reinforcement_learning_keras.enviroments.doom.doom_default_config import DoomDefaultConfig


class VizDoomPredictPositionConfig(DoomDefaultConfig):
    """Defines specific config for this env."""
    env_spec = 'VizdoomPredictPosition-v0'
    n_actions = 3
    res_scale = 0.2
    target_obs_shape = (90, 160)
