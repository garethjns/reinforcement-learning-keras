import numpy as np

from reinforcement_learning_keras.agents.q_learning.deep_q_agent import DeepQAgent
from reinforcement_learning_keras.enviroments.doom.vizdoom_basic_config import VizDoomBasicConfig
from reinforcement_learning_keras.enviroments.doom.vizdoom_corridor_config import VizDoomCorridorConfig
from reinforcement_learning_keras.enviroments.doom.vizdoom_deathmatch_config import VizDoomDeathmatchConfig
from reinforcement_learning_keras.enviroments.doom.vizdoom_defend_center_config import VizDoomDefendCenterConfig
from reinforcement_learning_keras.enviroments.doom.vizdoom_defend_line_config import VizDoomDefendLineConfig
from reinforcement_learning_keras.enviroments.doom.vizdoom_health_gathering_config import VizDoomHealthGatheringConfig
from reinforcement_learning_keras.enviroments.doom.vizdoom_health_gathering_supreme import \
    VizDoomHealthGatheringSupremeConfig
from reinforcement_learning_keras.enviroments.doom.vizdoom_my_way_home_config import VizDoomMyWayHomeConfig
from reinforcement_learning_keras.enviroments.doom.vizdoom_predict_position_config import VizDoomPredictPositionConfig
from reinforcement_learning_keras.enviroments.doom.vizdoom_take_cover_config import VizDoomTakeCoverConfig

if __name__ == "__main__":
    selected_config = VizDoomCorridorConfig

    available_configs = [VizDoomBasicConfig, VizDoomCorridorConfig, VizDoomDefendCenterConfig, VizDoomDefendLineConfig,
                         VizDoomHealthGatheringConfig, VizDoomMyWayHomeConfig, VizDoomPredictPositionConfig,
                         VizDoomTakeCoverConfig, VizDoomDeathmatchConfig, VizDoomHealthGatheringSupremeConfig]

    if selected_config is None:
        selected_config = np.random.choice(available_configs)

    agent = DeepQAgent.example(config=selected_config(agent_type='double_dueling_dqn', mode='stack'), n_episodes=20000,
                               max_episode_steps=20000, render=True, update_every=1, checkpoint_every=1000)
    agent.save()
