import warnings

try:
    from rlk.environments.doom.vizdoom_basic_config import VizDoomBasicConfig
    from rlk.environments.doom.vizdoom_corridor_config import VizDoomCorridorConfig
    from rlk.environments.doom.vizdoom_deathmatch_config import VizDoomDeathmatchConfig
    from rlk.environments.doom.vizdoom_defend_center_config import VizDoomDefendCenterConfig
    from rlk.environments.doom.vizdoom_defend_line_config import VizDoomDefendLineConfig
    from rlk.environments.doom.vizdoom_health_gathering_config import \
        VizDoomHealthGatheringConfig
    from rlk.environments.doom.vizdoom_health_gathering_supreme import \
        VizDoomHealthGatheringSupremeConfig
    from rlk.environments.doom.vizdoom_my_way_home_config import VizDoomMyWayHomeConfig
    from rlk.environments.doom.vizdoom_predict_position_config import \
        VizDoomPredictPositionConfig
    from rlk.environments.doom.vizdoom_take_cover_config import VizDoomTakeCoverConfig

    AVAILABLE = True
except ImportError:
    warnings.warn('vizdoomgym not installed.')
    AVAILABLE = False
