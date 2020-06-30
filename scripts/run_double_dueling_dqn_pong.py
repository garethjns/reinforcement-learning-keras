from reinforcement_learning_keras.agents.q_learning.deep_q_agent import DeepQAgent
from reinforcement_learning_keras.enviroments.atari.pong.pong_config import PongConfig


if __name__ == "__main__":
    agent = DeepQAgent.example(config=PongConfig(agent_type='double_dueling_dqn'), max_episode_steps=10000,
                               render=False, update_every=1, checkpoint_every=0)
    agent.save()