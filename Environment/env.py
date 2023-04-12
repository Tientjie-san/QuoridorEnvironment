"""
# Chess
```{figure} classic_chess.gif
:width: 140px
:name: chess
```
This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.
| Import             | `from pettingzoo.classic import chess_v5` |
|--------------------|------------------------------------|
| Actions            | Discrete                           |
| Parallel API       | Yes                                |
| Manual Control     | No                                 |
| Agents             | `agents= ['player_0', 'player_1']` |
| Agents             | 2                                  |
| Action Shape       | Discrete(4672)                     |
| Action Values      | Discrete(4672)                     |
| Observation Shape  | (8,8,20)                           |
| Observation Values | [0,1]                              |
Chess is one of the oldest studied games in AI. Our implementation of the observation and action spaces for chess are what the AlphaZero method uses, with two small changes.
### Observation Space
The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.
Like AlphaZero, the main observation space is an 8x8 image representing the board. It has 20 channels representing:
* Channels 0 - 3: Castling rights:
  * Channel 0: All ones if white can castle queenside
  * Channel 1: All ones if white can castle kingside
  * Channel 2: All ones if black can castle queenside
  * Channel 3: All ones if black can castle kingside
* Channel 4: Is black or white
* Channel 5: A move clock counting up to the 50 move rule. Represented by a single channel where the *n* th element in the flattened channel is set if there has been *n* moves
* Channel 6: All ones to help neural networks find board edges in padded convolutions
* Channel 7 - 18: One channel for each piece type and player color combination. For example, there is a specific channel that represents black knights. An index of this channel is set to 1 if a black knight is in the corresponding spot on the game board, otherwise, it is set to 0. En passant
possibilities are represented by displaying the vulnerable pawn on the 8th row instead of the 5th.
* Channel 19: represents whether a position has been seen before (whether a position is a 2-fold repetition)
Like AlphaZero, the board is always oriented towards the current agent (the currant agent's king starts on the 1st row). In other words, the two players are looking at mirror images of the board, not the same board.
Unlike AlphaZero, the observation space does not stack the observations previous moves by default. This can be accomplished using the `frame_stacking` argument of our wrapper.
#### Legal Actions Mask
The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.
### Action Space
From the AlphaZero chess paper:
> [In AlphaChessZero, the] action space is a 8x8x73 dimensional array.
Each of the 8×8 positions identifies the square from which to “pick up” a piece. The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7] in which the piece will be
moved, along one of eight relative compass directions {N, NE, E, SE, S, SW, W, NW}. The
next 8 planes encode possible knight moves for that piece. The final 9 planes encode possible
underpromotions for pawn moves or captures in two possible diagonals, to knight, bishop or
rook respectively. Other pawn moves or captures from the seventh rank are promoted to a
queen.
We instead flatten this into 8×8×73 = 4672 discrete action space.
You can get back the original (x,y,c) coordinates from the integer action `a` with the following expression: `(a/(8*73), (a/73)%8, a%(8*8))`
### Rewards
| Winner | Loser | Draw |
| :----: | :---: | :---: |
| +1     | -1    | 0 |
### Version History
* v5: Changed python-chess version to version 1.7 (1.13.1)
* v4: Changed observation space to proper AlphaZero style frame stacking (1.11.0)
* v3: Fixed bug in arbitrary calls to observe() (1.8.0)
* v2: Legal action mask in observation replaced illegal move list in infos (1.5.0)
* v1: Bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
* v0: Initial versions release (1.0.0)
"""


from quoridor import Quoridor
from pettingzoo.utils import AECEnv, wrappers


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

    def __init__(self):
        super().__init__()
        self.board = Quoridor()
        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]

        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
