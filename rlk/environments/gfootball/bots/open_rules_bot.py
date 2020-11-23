"""
This bot is modified from "best open rules bot score 1020.7" posted by Ken Miller
https://www.kaggle.com/mlconsult/best-open-rules-bot-score-1020-7

The agent follows the requirements set by the Kaggle competition (ie. is self contained and follows the below definition
of the agent function).

The main change is the addition of a decorator to get the raw observations from json on disk if they're not in obs.
This enables compatibility with the gym interface used by RLK.
"""

from typing import Any, Dict, Union

from kaggle_environments.envs.football.helpers import human_readable_agent, Action, GameMode

from rlk.environments.gfootball.bots.rlk_compatibility import rlk_compatibility

directions = [
    [Action.TopLeft, Action.Top, Action.TopRight],
    [Action.Left, Action.Idle, Action.Right],
    [Action.BottomLeft, Action.Bottom, Action.BottomRight]]

dirsign = lambda x: 1 if abs(x) < 0.01 else (0 if x < 0 else 2)

enemyGoal = [1, 0]
perfectRange = [[0.61, 1], [-0.2, 0.2]]


def inside(pos, area):
    return area[0][0] <= pos[0] <= area[0][1] and area[1][0] <= pos[1] <= area[1][1]


def get_distance(pos1, pos2):
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def player_direction(obs):
    controlled_player_pos = obs['left_team'][obs['active']]
    controlled_player_dir = obs['left_team_direction'][obs['active']]
    x = controlled_player_pos[0]
    y = controlled_player_pos[1]
    dx = controlled_player_dir[0]
    dy = controlled_player_dir[1]

    if x <= dx:
        return 0
    if x > dx:
        return 1


def run_pass(left_team, right_team, x, y):
    ###Are there defenders dead ahead?
    defenders = 0
    for i in range(len(right_team)):
        if right_team[i][0] > x and y + .01 >= right_team[i][1] and right_team[i][1] >= y - .01:
            if abs(right_team[i][0] - x) < .01:
                defenders = defenders + 1
    if defenders == 0:
        return Action.Right

    teammateL = 0
    teammateR = 0
    for i in range(len(left_team)):
        # is there a teamate close to left
        if left_team[i][0] >= x:
            if left_team[i][1] < y:
                if abs(left_team[i][1] - x) < .05:
                    teammateL = teammateL + 1

        # is there a teamate to right
        if left_team[i][0] >= x:
            if left_team[i][1] > y:
                if abs(left_team[i][1] - x) < .05:
                    teammateR = teammateR + 1
    # pass only close to goal
    if x > .75:
        if teammateL > 0 or teammateR > 0:
            return Action.ShortPass

    if defenders > 0 and y >= 0:
        return Action.TopRight

    if defenders > 0 and y < 0:
        return Action.BottomRight


@rlk_compatibility
@human_readable_agent
def agent(obs: Union[Dict[str, Any], None] = None):
    controlled_player_pos = obs['left_team'][obs['active']]

    # special plays
    if obs["game_mode"] == GameMode.Penalty:
        return Action.Shot
    if obs["game_mode"] == GameMode.Corner:
        if controlled_player_pos[0] > 0:
            return Action.Shot
    if obs["game_mode"] == GameMode.FreeKick:
        return Action.Shot

    # Make sure player is running.
    if 0 < controlled_player_pos[0] < 0.6 and Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint
    elif 0.6 < controlled_player_pos[0] and Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint

    # Does the player we control have the ball?
    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:

        goalkeeper = 0
        # if in the zone near goal shoot
        if inside(controlled_player_pos, perfectRange) and controlled_player_pos[0] < obs['ball'][0]:
            return Action.Shot
        # if the goalie is coming out on player near goal shoot
        elif abs(obs['right_team'][goalkeeper][0] - 1) > 0.2 and controlled_player_pos[0] > 0.4 and abs(
                controlled_player_pos[1]) < 0.2:
            return Action.Shot
        # if close to goal and too wide for shot pass the ball
        if controlled_player_pos[0] > .75 and controlled_player_pos[1] > .20 or controlled_player_pos[0] > .75 and \
                controlled_player_pos[1] < -.20:
            return Action.ShortPass
        # if near our goal and moving away long pass to get out of our zone
        if player_direction(obs) == 1 and controlled_player_pos[0] < -.3:
            return Action.LongPass
        # which way should we run or pass
        else:
            return run_pass(obs['left_team'], obs['right_team'], controlled_player_pos[0], controlled_player_pos[1])
    else:
        # vector where ball is going
        ball_targetx = obs['ball'][0] + obs['ball_direction'][0]
        ball_targety = obs['ball'][1] + obs['ball_direction'][1]

        # euclidian distance to the ball so we head off movement until very close
        e_dist = get_distance(obs['left_team'][obs['active']], obs['ball'])

        # if not close to ball move to where it is going
        if e_dist > .005:
            # Run where ball will be
            xdir = dirsign(ball_targetx - controlled_player_pos[0])
            ydir = dirsign(ball_targety - controlled_player_pos[1])
            return directions[ydir][xdir]
        # if close to ball go to ball
        else:
            # Run towards the ball.
            xdir = dirsign(obs['ball'][0] - controlled_player_pos[0])
            ydir = dirsign(obs['ball'][1] - controlled_player_pos[1])
            return directions[ydir][xdir]
