from quoridor import Quoridor
from pettingzoo.utils import AECEnv, wrappers
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
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(209,), dtype=np.int8
                    ),
                }
            )
            for name in self.agents
        }

        self.render_moded = render_mode

        # these are mandatory for the AEC API
        self.render_mode = render_mode
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.infos = {
            name: {"pgn": self.board.get_pgn(), "turn": len(self.board.moves) + 1}
            for name in self.agents
        }
        self.agent_selection = None
        self.rewards = None
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}

    def reset(self, seed=None, return_info=False, options=None):
        self.has_reset = True

        self.agents = self.possible_agents[:]

        self.board = Quoridor()

        self._agent_selector = agent_selector(self.agents)

        # Mandatrory for AEC API
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {
            name: {"pgn": self.board.get_pgn(), "turn": len(self.board.moves) + 1}
            for name in self.agents
        }

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        chosen_move = convet_discrete_to_quoridor_move(action)
        self.board.make_move(chosen_move)
        game_over = self.board.is_terminated

        if game_over:
            # the game reward value is in perspective of the first agent.
            if self.agent_selection == self.agents[0]:
                self.set_game_result(1)
            else:
                self.set_game_result(-1)

        self._accumulate_rewards()
        for name in self.agents:
            self.infos[name] = {
                "pgn": self.board.get_pgn(),
                "turn": len(self.board.moves) + 1,
            }

        self.agent_selection = (
            self._agent_selector.next()
        )  # Give turn to the next agent

        if self.render_mode == "human":
            self.render()

    def set_game_result(self, result_val):
        for i, name in enumerate(self.agents):
            self.terminations[name] = True
            # the winning agent gets a reward of 1, the losing agent gets the negative of that reward
            result_coef = 1 if i == 0 else -1
            self.rewards[name] = result_val * result_coef
            self.infos[name] = {
                "legal_moves": [],
                "pgn": self.board.get_pgn(),
                "turn": len(self.board.moves),
            }

    def observe(self, agent):
        observation = board_to_observation(self.board)
        action_mask = np.zeros(209, dtype=np.int8)
        for move in self.board.get_legal_moves():
            action_mask[convert_quoridor_move_to_discrete(move)] = True
        return {"observation": observation, "action_mask": action_mask}

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def close(self):
        # mandatory for AEC API when render is defined
        pass
