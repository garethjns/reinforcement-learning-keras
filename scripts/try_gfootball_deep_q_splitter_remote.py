from rlk.agents.q_learning.deep_q_agent import DeepQAgent
from rlk.environments.gfootball.gfootball_config import GFootballConfig

if __name__ == "__main__":
    agent = DeepQAgent(**GFootballConfig('double_dueling_dqn', remote=True,
                                         env_spec="GFootball-academy_empty_goal_close-v0").build(),
                       env_builder_kwargs={'remote': True, 'ip': '192.168.68.124'})

    agent.train(verbose=True, render=False,
                n_episodes=1000, max_episode_steps=10000, update_every=10,
                checkpoint_every=10)
