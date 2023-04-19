from gymnasium.spaces import Box, Discrete
from quoridor import Quoridor
from Environment.utils import (
    board_to_observation,
    convet_discrete_to_quoridor_move,
    convert_quoridor_move_to_discrete,
)
import numpy as np


class SimpleQuordiorEnv:
    def __init__(self):
        self.board = Quoridor()
        self.observation_space = Box(low=0, high=1, shape=(9, 9, 6), dtype=bool)
        self.action_space = Discrete(209)

    def reset(self):
        ...

    def step(self, action):
        ...

    def render(self):
        ...

    def observe(self):
        """Returns the observation of the current state.

        Returns
        -------
        observation : dict[str, np.ndarray]
            The observation of the current state.
        """
        observation = board_to_observation(self.board)
        action_mask = np.zeros(209, dtype=bool)
        for move in self.board.get_legal_moves():
            action_mask[convert_quoridor_move_to_discrete(move)] = True
        return {"observation": observation, "action_mask": action_mask}
