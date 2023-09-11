from rlgym_sim.utils import TerminalCondition
from rlgym_sim.utils.gamestates import GameState

RAMP_HEIGHT = 256

class AerialTaskTerminalCondition(TerminalCondition):

    def __init__(self):
        self._has_flown = False

    def reset(self, state: GameState):
        self._has_flown = False

    def is_terminal(self, state: GameState) -> bool:
        if not self._ball_has_flown(state):
            return False
        else:
            return self._ball_has_landed(state)

    def _ball_has_flown(self, state: GameState) -> bool:
        if self._has_flown:
            return True

        if state.ball.position[2] < RAMP_HEIGHT:
            return False

        for player in state.players:
            if player.ball_touched:
                self._has_flown = True
                return True

    def _ball_has_landed(self, state):
        return state.ball.position[2] < RAMP_HEIGHT
