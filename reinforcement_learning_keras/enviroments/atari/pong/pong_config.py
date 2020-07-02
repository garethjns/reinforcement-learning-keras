from reinforcement_learning_keras.enviroments.atari.atari_default_config import AtariDefaultConfig


class PongConfig(AtariDefaultConfig):
    """Defines configs tweaks for Pong."""
    env_spec = 'PongNoFrameskip-v4'
