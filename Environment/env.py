"""
# Quoridor Environment


| Import             | `from Environment import QuoridorEnv, env` |
|--------------------|------------------------------------|
| Actions            | Discrete                           |
| Parallel API       | No                                |
| Manual Control     | No                                 |
| Agents             | `agents= ['player_1', 'player_2']` |
| Agents             | 2                                  |
| Action Shape       | Discrete(209)                     |
| Action Values      | Discrete(209)                     |
| Observation Shape  | (9,9,6)                           |
| Observation Values | [0,1]                              |


Quoridor is a two-player strategy board game that was invented by Mirko Marchesi in 1991. 
The game is played on a square board with a 9x9 grid of squares. 
Each player starts with a pawn on opposite sides of the board, 
and the objective is to be the first player to reach the opposite side of the board with their pawn.

Players can also place walls on the board to block their opponent's pawn from moving forward. 
The game is simple to learn, but it involves complex strategic decisions and 
can be challenging to master. 
Quoridor is or will become popular in the world of artificial intelligence and game theory 
as a testbed for developing new algorithms and strategies.

### Observation Space

The observation is a dictionary which contains an `'observation'` element 
which is the usual RL observation described below, 
and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.

the main observation space is an 9x9 image representing the board. It has 6 channels representing:

* Channels 0: position of player 1's pawn: 1 if player 1's pawn is in this position (0 otherwise)
* Channel 1: position of player 2's pawn: 1 if player 2's pawn is in this position (0 otherwise)
* Channel 2: position of player 1's walls: 1 if player 1's wall is in this position (0 otherwise)
* Channel 3: position of player 2's walls: 1 if player 2's wall is in this position (0 otherwise)
* Channel 4: total number of walls left for player 1 (count all 1s)
* Channel 5: total number of walls left for player 2 (count all 1s)

#### Legal Actions Mask

The legal moves available to the current agent are found in the `action_mask` element of 
the dictionary observation. 
The `action_mask` is a binary vector where each index of the vector represents whether the action is 
legal or not. 
The `action_mask` will be all zeros for any agent except the one whose turn it is. 
Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and 
a reward of 0 for all other agents.

### Action Space

Action Space
In quoridor the amount of possible pawn moves in the game is equal to 
the amount of cells of the board. 
In this environment we play the traditional quoridor variant, 1vs1. So the board size is 9x9. 
Which means that there are 81 possible pawn moves. 
The walls on the other hand can’t be placed on edges of the board so 
the walls can be placed in a space of 8x8. 
Since you can place a wall horizontally or vertically,
the total amount of wall moves = 8x8x2 = 128 wall moves. 
Which means the action space can be represented with an action shape of Discrete(209). 

| Indexes            | Type of moves |
|--------------------|------------------------------------|
| 0..80              | represent the pawn moves           |
| 81..144            | represent the horizontal wall moves
| 145..208           | represent the vertical wall moves  |


### Rewards

| Winner | Loser | Draw |
| :----: | :---: | :---: |
| +1     | -1    | 0 |
"""


from quoridor import Quoridor
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import AECEnv, wrappers
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.test import api_test  # noqa: E402
from Environment.utils import (
    board_to_observation,
    convert_discrete_to_quoridor_move,
    convert_quoridor_move_to_discrete,
)


def env(render_mode=None):
    """Create a Quoridor environment"""
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
        self.has_reset = False

    def reset(
        self,
        seed=None,
        options=None,
        return_info=False,
    ):
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
        """
        Step forward in the environment with the given action
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        chosen_move = convert_discrete_to_quoridor_move(action)
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
        """
        update the rewards and terminations for the agents based on the game result
        """
        for i, name in enumerate(self.agents):
            self.terminations[name] = True
            # the winning agent gets a reward of 1, the losing agent gets the negative reward
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


if __name__ == "__main__":
    api_test(QuoridorEnv(), num_cycles=1_000_000)
