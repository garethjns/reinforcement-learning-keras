from rlk.agents.q_learning.deep_q_agent import DeepQAgent
from rlk.environments.atari.pong.pong_config import PongConfig

if __name__ == "__main__":
    agent = DeepQAgent.example(config=PongConfig(agent_type='dqn', mode='stack'),
                               max_episode_steps=10000,
                               render=True, update_every=1, checkpoint_every=0)
    agent.save()
