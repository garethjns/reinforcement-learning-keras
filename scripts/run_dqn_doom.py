from reinforcement_learning_keras.agents.q_learning.deep_q_agent import DeepQAgent
from reinforcement_learning_keras.enviroments.doom.doom_config import DoomConfig

if __name__ == "__main__":
    agent = DeepQAgent.example(config=DoomConfig(agent_type='dqn', mode='stack'), n_episodes=1000,
                               max_episode_steps=10000, render=False, update_every=1, checkpoint_every=250)
    agent.save()
