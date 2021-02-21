from rlk.agents.actor_critic.actor_critic import ActorCriticAgent
from rlk.environments.cart_pole.cart_pole_config import CartPoleConfig

if __name__ == "__main__":
    agent = ActorCriticAgent.example(config=CartPoleConfig(agent_type='actor_critic'), max_episode_steps=500,
                                     n_episodes=2000,
                                     render=False,
                                     update_every=6, checkpoint_every=0)
    agent.save()
