from typing import List, Tuple, Optional

import numpy as np

from rlk.agents.components.helpers.env_builder import EnvBuilder
from rlk.agents.q_learning.deep_q_agent import DeepQAgent
from rlk.environments.doom import VizDoomCorridorConfig
from rlk.environments.doom.bots.scripted_bot import ScriptedBot


class World:
    """
    Handles:
        - Agent
        - ReplayBuffer (via agent)
        - Exploration (via agent)
        - Experience collection from additional policies

    First version:
     - Aims to use existing agent API as is, which leaves most responsibility inside the agent.
     - Provides place to abstract potentially common elements; replay buffer, env building, training logic, etc.
     - New functionality (additional experience collection) will focus on ScriptedBots for VizDoom envs.
     - Using DQNs

    Structure
       - Will use the existing agent for most things - (ie. the replay buffer will be the agent's)
       - Additional experience will be added to the agent's replay buffer
       - Most train logic still in agent

    Train loop:
      - agent.train()
      - collect additional experience
      - Add additional experience to replay buffer

    Possible later versions:
        - Handle "distribution" (joblib locally)?
        - Handle agent construction?
        - Common replay buffer?
        - Multiple additional experience collection policies?
        - Handles wrappers with bots
    """

    def __init__(self, env_builder: EnvBuilder, agent: DeepQAgent, experience_collectors: List[ScriptedBot],
                 max_episode_steps: int = 20000, n_jobs: int = 5):
        self.env_builder = env_builder
        self.agent = agent
        self.experience_collectors = experience_collectors
        self.max_episode_steps = max_episode_steps

    @staticmethod
    def _collect_additional_experience(bot: ScriptedBot, env_builder: EnvBuilder,
                                       max_steps: int) -> Tuple[List[np.ndarray], List[int], List[float], List[bool]]:
        env = env_builder.env
        env.reset()
        bot.reset()
        step = 0
        rewards = []
        obs = []
        dones = []
        actions = []

        d = False
        while not d:
            if step > max_steps:
                break
            a = bot.get_action()
            s, r, d, _ = env.step(a)

            rewards.append(r)
            obs.append(s)
            dones.append(d)
            actions.append(a)

            step += 1

        return obs, actions, rewards, dones

    def _train_step(self, agent_train_steps: int, render: bool, checkpoint_every: int = 0):
        # Run a train episode for the agent
        self.agent.train(n_episodes=agent_train_steps, max_episode_steps=20000, render=render, update_every=1,
                         checkpoint_every=checkpoint_every)

        # Collect additional experience
        for bot in self.experience_collectors:
            ss, aa, rr, dd = self._collect_additional_experience(bot=bot, env_builder=self.env_builder,
                                                                 max_steps=self.max_episode_steps)
            print(f"Bot scored {np.round(np.sum(rr))}")

            for s, a, r, d in zip(ss, aa, rr, dd):
                self.agent.replay_buffer.append((s, a, r, d))

    def train(self, n_train_steps: int = 10000, render: bool = False, agent_train_steps: Optional[int] = None,
              checkpoint_every: Optional[int] = None):

        if agent_train_steps is None:
            agent_train_steps = 0 + max(1, len(self.experience_collectors))

        if checkpoint_every is None:
            checkpoint_every = agent_train_steps

        for _ in range(n_train_steps):
            self._train_step(agent_train_steps=agent_train_steps, render=render, checkpoint_every=checkpoint_every)


if __name__ == "__main__":
    agent_config = VizDoomCorridorConfig(agent_type='double_dueling_dqn', mode='stack').build()
    # agent_ = DeepQAgent(**agent_config)

    agent_ = DeepQAgent.load("DoubleDuelingDQN_VizdoomCorridor-v0")

    world = World(
        agent=agent_,
        env_builder=EnvBuilder(env_spec=agent_config['env_spec'], env_wrappers=agent_config["env_wrappers"]),
        experience_collectors=[
            ScriptedBot(agent_config['env_spec'], script_n=sn) for sn in range(4)
        ]
    )

    world.train(n_train_steps=1000, agent_train_steps=None)

    agent_.play_episode(training=False, render=True, max_episode_steps=6000)
