from rlk.environments.doom.doom_default_config import DoomDefaultConfig


class VizDoomDefendCenterConfig(DoomDefaultConfig):
    """Defines specific config for this env."""
    env_spec = 'VizdoomDefendCenter-v0'
    n_actions = 3
    res_scale = 0.2
