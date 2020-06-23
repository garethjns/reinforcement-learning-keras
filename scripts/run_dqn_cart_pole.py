from agents.q_learning.deep_q_agent import DeepQAgent
from enviroments.cart_pole.cart_pole_config import CartPoleConfig

if __name__ == "__main__":
    agent = DeepQAgent.example(config=CartPoleConfig(agent_type='double_dqn'), max_episode_steps=10000, render=False,
                               update_every=6, checkpoint_every=0)
    agent.save()
