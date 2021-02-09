from rlk.agents.q_learning.deep_q_agent import DeepQAgent
from rlk.environments.cart_pole.cart_pole_config import CartPoleConfig

if __name__ == "__main__":
    agent = DeepQAgent.example(config=CartPoleConfig(agent_type='dqn'), max_episode_steps=500, n_episodes=2000,
                               render=False, update_every=6, checkpoint_every=0)
    agent.save()
