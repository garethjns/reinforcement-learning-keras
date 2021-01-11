import numpy as np
from joblib import Parallel, delayed

from rlk.environments.doom.bots.script_library import SCRIPTS
from rlk.environments.doom.bots.scripted_bot import ScriptedBot
from scripts.doom.run_scripted_bot_doom import run_bot

POOL = Parallel(n_jobs=20)


def eval_script(env_name: str, script_n: int, n: int = 100):
    bot = ScriptedBot(env_name, script_n)

    jobs = (delayed(run_bot)(env_name, bot, False) for _ in range(n))
    total_rewards = POOL(jobs)

    return np.mean(total_rewards), np.std(total_rewards)


if __name__ == "__main__":
    env_name = 'VizdoomCorridor-v0'
    n_reps = 100
    for sn in range(len(SCRIPTS[env_name].keys())):
        total_reward_mean, total_reward_std = eval_script(env_name, sn, n=n_reps)
        print(f"Script: {sn}: Mean reward ({n_reps} runs): "
              f"{np.round(total_reward_mean)} ± {np.round(total_reward_std)}")
