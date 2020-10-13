from reinforcement_learning_keras.environments.doom.doom_default_config import DoomDefaultConfig


class VizDoomHealthGatheringConfig(DoomDefaultConfig):
    """Defines specific config for this env."""
    env_spec = 'VizdoomHealthGathering-v0'
    n_actions = 3
