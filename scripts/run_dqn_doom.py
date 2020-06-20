from agents.q_learning.deep_q_agent import DeepQAgent
from enviroments.doom.doom_config import DoomConfig

if __name__ == "__main__":
    agent = DeepQAgent.example(config=DoomConfig(agent_type='dqn'), n_episodes=1000,
                               render=True,
                               checkpoint_every=0)
    agent.save()
