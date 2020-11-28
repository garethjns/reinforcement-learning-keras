"""
Download historical games from the Kaggle API and create a dataset.

Games are downloaded as json including the raw observations at each step. These are zipped and saved.

After downloading the games are collected into a .hdf file. The features are generated from the json to create
structured data of the simple, raw, and smm obs. SMM is turned off for now.

This dataset is structured as:
[features] -> action taken
"""

import json

from kaggle_football.api.leaderboard import LeaderBoard

TOP_TEAMS = 2
EPISODE_MIN_SCORE = 1580

if __name__ == "__main__":
    lb = LeaderBoard(team_min_rank=TOP_TEAMS, episode_min_score=EPISODE_MIN_SCORE,
                     n_download_jobs=2, n_process_jobs=28)
    lb.using('episode_repo/', get_smm=False)
    lb.get_top_teams()  # May be slow, downloads ~600mb. Can also use pre-downloaded version:
    lb.download_episodes()  # May fail randomly
    lb.collect(episode_min_score=EPISODE_MIN_SCORE)
    lb.save(fn=f"downloaded_games")
