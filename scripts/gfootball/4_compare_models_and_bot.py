"""
This script compares the pretrained model, rl model, and the open rules bot used during RL training.
"""

import sys
from typing import Tuple, Dict, List, Any

import numpy as np
from joblib import Parallel, delayed
from kaggle_environments import make

sys.path.append("scripts/gfootball")
N_GAMES = 2
N_JOBS = 2


def play(left_player, right_player, print_details=True,
         save_video=False, debug=True) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    env = make("football", debug=debug,
               configuration={"save_video": save_video,
                              "scenario_name": "11_vs_11_kaggle"})

    output = env.run([left_player, right_player])

    if print_details:
        for s, (left, right) in enumerate(output):
            print(f"\nStep {s}")

            print(f"Left player ({left_player}): \n"
                  f"actions taken: {left['action']}, "
                  f"reward: {left['reward']}, "
                  f"status: {left['status']}, "
                  f"info: {left['info']}")

            print(f"Right player ({right_player}): \n"
                  f"actions taken: {right['action']}, "
                  f"reward: {right['reward']}, "
                  f"status: {right['status']}, "
                  f"info: {right['info']}\n")

        print(f"Final score: {output[-1][0]['reward']} : {output[-1][1]['reward']}")

    return output


def play_multiple(left_player, right_player, print_details=False, save_video=False, debug=True):
    games = Parallel(n_jobs=N_JOBS)(delayed(play)(left_player, right_player, print_details=print_details, debug=debug,
                                                  save_video=save_video)
                                    for _ in range(N_GAMES))
    print(f"\n {left_player} vs {right_player}:")
    left_score = []
    for game in games:
        if (game[-1][0]['reward'] is not None) and (game[-1][1]['reward'] is not None):
            left_score.append(game[-1][0]['reward'])
        print(f"Final score: {game[-1][0]['reward']} : {game[-1][1]['reward']}")

    return np.mean(left_score), np.std(left_score)


if __name__ == "__main__":

    # Assuming running from top level of repo
    base_path = 'scripts/gfootball/'
    bot_path = 'rlk/environments/gfootball/bots/open_rules_bot.py'

    comparisons = [
        (f"{base_path}kaggle_agent_classification_model.py", f"{base_path}kaggle_agent_rl_model.py")]
    # TODO: (f"{base_path}kaggle_agent_classification_model.py", bot_path),
    # TODO: (f"{base_path}kaggle_agent_rl.py", bot_path)]

    for left_player, right_player in comparisons:
        average_score, std_score = play_multiple(left_player=left_player,
                                                 right_player=right_player)

        print(f"\n {left_player} vs {right_player}:\n"
              f"Left agent mean +/- std goal diff {average_score} +/- {std_score}")
