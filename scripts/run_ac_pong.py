from rlk.agents.actor_critic.actor_critic import ActorCriticAgent
from rlk.environments.atari.pong.pong_config import PongConfig

if __name__ == "__main__":
    agent = ActorCriticAgent.example(config=PongConfig(agent_type='actor_critic', mode='stack'), max_episode_steps=20000,
                                     n_episodes=2000,
                                     render=True, checkpoint_every=0)
    agent.save()
