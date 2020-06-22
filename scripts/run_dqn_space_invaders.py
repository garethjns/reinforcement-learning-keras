from agents.q_learning.deep_q_agent import DeepQAgent
from enviroments.atari.space_invaders.space_invaders_config import SpaceInvadersConfig


if __name__ == "__main__":
    agent = DeepQAgent.example(config=SpaceInvadersConfig(agent_type='dqn'), max_episode_steps=10000, render=True,
                               update_every=6, checkpoint_every=0)
    agent.save()
