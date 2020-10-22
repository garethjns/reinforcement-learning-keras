from rlk.agents.q_learning.deep_q_agent import DeepQAgent
from rlk.environments.cart_pole.cart_pole_config import CartPoleConfig

if __name__ == "__main__":
    agent = DeepQAgent.example(config=CartPoleConfig(agent_type='double_dqn'), max_episode_steps=250, n_episodes=100,
                               render=False,
                               update_every=3, checkpoint_every=0)
    agent.save()
