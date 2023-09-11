import numpy as np

from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.common_values import BACK_WALL_Y
from rlgym_sim.utils.gamestates import PlayerData, GameState
from typing import Optional

RAMP_HEIGHT = 256
UP_VECTOR = np.array([0.0, 0.0, 1.0])
G_CONST = np.array([0.0, 0.0, -650.0])

class AerialDistanceReward(RewardFunction):
    def __init__(self, height_scale: float = 10, distance_scale: float = 10, scale_by_upness: bool = False, tick_skip: int = 0):
        super().__init__()
        self.height_scale = height_scale
        self.distance_scale = distance_scale

        self.current_car: Optional[PlayerData] = None
        self.prev_state: Optional[GameState] = None
        self.ball_distance: float = 0
        self.car_distance: float = 0
        self.scale_by_upness = scale_by_upness
        self.tick_skip = tick_skip

        if scale_by_upness and tick_skip == 0:
            raise ValueError("Must assign tick_skip when using scale_by_upness")

    def reset(self, initial_state: GameState):
        self.current_car = None
        self.prev_state = initial_state

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = 0
        is_current = self.current_car is not None and self.current_car.car_id == player.car_id
        # Test if player is on the ground
        if player.car_data.position[2] < RAMP_HEIGHT:
            if is_current:
                is_current = False
                self.current_car = None
        # First non ground touch detection
        elif player.ball_touched and not is_current:
            is_current = True
            self.ball_distance = 0
            self.car_distance = 0
            upness = 1

            if self.scale_by_upness:
                delta_v_due_to_gravity = G_CONST * self.tick_skip / 120.0
                delta_v = self.prev_state.ball.linear_velocity - state.ball.linear_velocity - delta_v_due_to_gravity
                accel_vector = delta_v / np.linalg.norm(delta_v)
                upness = np.dot(accel_vector, UP_VECTOR)

            rew = upness * self.height_scale * max(player.car_data.position[2] + state.ball.position[2] - 2 * RAMP_HEIGHT, 0)

        # Still off the ground after a touch, add distance and reward for more touches
        elif is_current:
            self.car_distance += np.linalg.norm(player.car_data.position[0:2] - self.current_car.car_data.position[0:2])
            self.ball_distance += np.linalg.norm(state.ball.position[0:2] - self.prev_state.ball.position[0:2])
            # Cash out on touches
            if player.ball_touched:
                upness = 1

                if self.scale_by_upness:
                    delta_v_due_to_gravity = G_CONST * self.tick_skip / 120.0
                    delta_v = self.prev_state.ball.linear_velocity - state.ball.linear_velocity - delta_v_due_to_gravity
                    accel_vector = delta_v / np.linalg.norm(delta_v)
                    upness = np.dot(accel_vector, UP_VECTOR)

                    # make "upness" be bounded by [0, 1]
                    upness = np.clip(upness, 0, 0.8) * 1.25

                rew = upness * self.distance_scale * (self.car_distance + self.ball_distance)
                self.car_distance = 0
                self.ball_distance = 0

        if is_current:
            self.current_car = player  # Update to get latest physics info

        self.prev_state = state

        return rew / (2 * BACK_WALL_Y)
