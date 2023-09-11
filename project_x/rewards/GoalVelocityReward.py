import numpy as np

from typing import List

from rlgym_sim.utils import RewardFunction, math
from rlgym_sim.utils.common_values import (
    BLUE_TEAM,
    ORANGE_TEAM,
    ORANGE_GOAL_BACK,
    BLUE_GOAL_BACK,
    BALL_MAX_SPEED,
    CAR_MAX_SPEED,
)
from rlgym_sim.utils.gamestates import GameState, PlayerData


def same_team(*args: List[PlayerData]):
    return len(args) == 0 or all(player.team_num == args[0].team_num for player in args)


def same_player(*args: List[PlayerData]):
    return len(args) == 1 or (
        len(args) > 1 and all(player.car_id == args[0].car_id for player in args)
    )


class GoalVelocityReward(RewardFunction):
    """
    a goal reward that scales with the speed of the ball as it crosses into the goal
    """

    def __init__(self):
        super().__init__()

        # Need to keep track of last registered value to detect changes
        self._goals = {}
        self._state = None
        self._goal_speed = 0

    def reset(self, initial_state: GameState, optional_data=None):
        self._state = None

    def pre_step(self, state: GameState):
        self._goal_speed = np.linalg.norm(self._state.ball.linear_velocity)
        self._goals = {
            player.car_id: player.match_goals for player in self._state.players
        }

    def get_reward(
        self,
        player: PlayerData,
        state: GameState,
        previous_action: np.ndarray,
        optional_data=None,
    ):
        scorer = self._who_scored(state.players)

        if scorer is None:
            return 0

        # you don't get rewarded for how fast your teammates send the ball into the goal
        if not same_player(scorer, player) and same_team(scorer, player):
            return 0

        reward_coeff = 1 if same_player(player, scorer) else -1

        reward = reward_coeff * self._goal_speed / BALL_MAX_SPEED

        return reward

    def _who_scored(self, players: List[PlayerData]):
        if len(self._goals) == 0:
            return None

        scorer = [
            player
            for player in players
            if player.match_goals > self._goals[player.car_id]
        ]
        return scorer[0] if scorer else None
