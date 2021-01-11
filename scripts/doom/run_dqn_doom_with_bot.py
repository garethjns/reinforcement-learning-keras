from rlk.agents.components.helpers.env_builder import EnvBuilder
from rlk.agents.q_learning.deep_q_agent import DeepQAgent
from rlk.environments.doom import VizDoomCorridorConfig
from rlk.environments.doom.bots.scripted_bot import ScriptedBot
from rlk.world.world import World

if __name__ == "__main__":
    agent_config = VizDoomCorridorConfig(agent_type='double_dueling_dqn', mode='stack').build()
    agent_ = DeepQAgent(**agent_config)
    # Or load with agent_ = DeepQAgent(**agent_config)

    # World defines the agent, environment, and bots.
    world = World(
        agent=agent_,
        env_builder=EnvBuilder(env_spec=agent_config['env_spec'], env_wrappers=agent_config["env_wrappers"]),
        experience_collectors=[
            # There are currently 4 scripted bots available for VizDoomCorridor
            ScriptedBot(agent_config['env_spec'], script_n=0),
            ScriptedBot(agent_config['env_spec'], script_n=1),
            ScriptedBot(agent_config['env_spec'], script_n=2),
            ScriptedBot(agent_config['env_spec'], script_n=3)
        ]
    )

    # Train
    world.train(n_train_steps=10000, agent_train_steps=None)

    # Play and render single episode
    agent_.play_episode(training=False, render=True, max_episode_steps=6000)
