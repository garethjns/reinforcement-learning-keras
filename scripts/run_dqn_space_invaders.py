from rlk.agents.q_learning.deep_q_agent import DeepQAgent
from rlk.environments.atari.space_invaders.space_invaders_config import SpaceInvadersConfig

if __name__ == "__main__":
    agent = DeepQAgent.example(config=SpaceInvadersConfig(agent_type='dueling_dqn', mode='stack'),
                               max_episode_steps=4000, n_episodes=10000,
                               render=False, update_every=2, checkpoint_every=2000)
    agent.save()
