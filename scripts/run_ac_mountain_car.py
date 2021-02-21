from rlk.agents.actor_critic.actor_critic import ActorCriticAgent
from rlk.environments.mountain_car.mountain_car_config import MountainCarConfig

if __name__ == "__main__":
    agent = ActorCriticAgent.example(config=MountainCarConfig(agent_type='actor_critic'), max_episode_steps=500,
                                     n_episodes=1000, render=True,  checkpoint_every=0)
    agent.save()
