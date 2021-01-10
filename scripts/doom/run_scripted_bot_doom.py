import time

import gym

from rlk.environments.doom.bots.scripted_bot import ScriptedBot
from rlk.environments.doom.vizdoom_corridor_config import VizDoomCorridorConfig  # noqa


def run_bot(env_name: str, bot: ScriptedBot, render: bool = True, max_steps: int = 20000):
    env = gym.make(env_name)
    env.reset()
    bot.reset()
    step = 0
    rewards = []
    done = False

    while not done:
        if step > max_steps:
            break
        _, r, done, _ = env.step(bot.get_action())
        rewards.append(r)

        if render:
            env.render()
            time.sleep(0.01)

        step += 1

    return sum(rewards)


if __name__ == "__main__":
    env_name_ = 'VizdoomCorridor-v0'
    bot_ = ScriptedBot(env_name_, 0)

    run_bot(env_name_, bot_)
