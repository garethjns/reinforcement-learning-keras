"""
Using create_env from gfootball - adds some wrappers.

Alternative using football env:
academy_3_vs_1_with_keeper_env = lambda: FootballEnv(Config(values={'level': "academy_3_vs_1_with_keeper"}))
"""

import gym

try:
    from gfootball.env import create_environment
except ImportError:
    def create_environment(*args, **kwargs):
        pass

COMMON_KWARGS = {"stacked": False, "representation": 'raw', "write_goal_dumps": False,
                 "write_full_episode_dumps": False, "write_video": False, "render": False,
                 "rewards": 'scoring,checkpoints',
                 "number_of_left_players_agent_controls": 1, "number_of_right_players_agent_controls": 0}

SUPPORTED_ENVS = ["kaggle_11_vs_11", "academy_empty_goal_close", "academy_empty_goal", "academy_run_to_score",
                  "academy_pass_and_shoot_with_keeper", "academy_run_pass_and_shoot_with_keeper",
                  "academy_run_to_score_with_keeper", "academy_counterattack_easy", "academy_3_vs_1_with_keeper"]


def kaggle_11_vs_11():
    return create_environment(env_name='11_vs_11_kaggle', **COMMON_KWARGS)


def academy_empty_goal_close():
    return create_environment(env_name='academy_empty_goal_close', **COMMON_KWARGS)


def academy_empty_goal():
    return create_environment(env_name='academy_empty_goal', **COMMON_KWARGS)


def academy_run_to_score():
    return create_environment(env_name='academy_run_to_score', **COMMON_KWARGS)


def academy_pass_and_shoot_with_keeper():
    return create_environment(env_name='academy_pass_and_shoot_with_keeper', **COMMON_KWARGS)


def academy_run_pass_and_shoot_with_keeper():
    return create_environment(env_name='academy_run_pass_and_shoot_with_keeper', **COMMON_KWARGS)


def academy_run_to_score_with_keeper():
    return create_environment(env_name='academy_run_to_score_with_keeper', **COMMON_KWARGS)


def academy_counterattack_easy():
    return create_environment(env_name='academy_counterattack_easy', **COMMON_KWARGS)


def academy_3_vs_1_with_keeper():
    return create_environment(env_name='academy_3_vs_1_with_keeper', **COMMON_KWARGS)


def register_all():
    for env_name in SUPPORTED_ENVS:
        try:
            gym.envs.register(id=f"GFootball-{env_name}-v0", max_episode_steps=10000,
                              entry_point=f"rlk.environments.gfootball.register_environments:{env_name}")
        except gym.error.Error:
            pass
