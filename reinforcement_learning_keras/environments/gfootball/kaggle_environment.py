from gfootball.env import create_environment

kaggle_environment = lambda: create_environment(env_name='11_vs_11_kaggle',
                                                stacked=False,
                                                representation='raw',
                                                write_goal_dumps=False,
                                                write_full_episode_dumps=False,
                                                write_video=False,
                                                render=False,
                                                number_of_left_players_agent_controls=1,
                                                number_of_right_players_agent_controls=0)
