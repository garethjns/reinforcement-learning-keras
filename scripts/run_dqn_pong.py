from agents.q_learning.deep_q_agent import DeepQAgent
from enviroments.pong.pong_config import PongConfig


if __name__ == "__main__":
    agent = DeepQAgent.example(config=PongConfig(agent_type='dqn'), max_episode_steps=10000, render=False,
                               update_every=6, checkpoint_every=100)
    agent.save()
