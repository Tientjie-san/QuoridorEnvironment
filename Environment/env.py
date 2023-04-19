from quoridor import Quoridor
from pettingzoo.utils import AECEnv, wrappers
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from Environment.utils import (
    board_to_observation,
    convet_discrete_to_quoridor_move,
    convert_quoridor_move_to_discrete,
)


def env(render_mode=None):
    env = QuoridorEnv(
        render_mode=render_mode,
    )
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class QuoridorEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "name": "quoridor_v0",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.board = Quoridor()
        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.action_spaces = {name: spaces.Discrete(209) for name in self.agents}
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(9, 9, 6), dtype=bool
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(209,), dtype=bool),
                }
            )
            for name in self.agents
        }

        self.render_moded = render_mode

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def observe(self, agent):
        observation = board_to_observation(self.board)
        action_mask = np.zeros(209, dtype=bool)
        for move in self.board.get_legal_moves():
            action_mask[convert_quoridor_move_to_discrete(move)] = True
        return {"observation": observation, "action_mask": action_mask}

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
