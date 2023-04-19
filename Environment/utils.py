from typing import Dict, List, Tuple
import numpy as np
from quoridor import Quoridor


def convet_discrete_to_quoridor_move(discrete_move: int) -> str:
    """
    Converts a discrete move Discrete(209) to a quoridor move.

    Parameters
    ----------
    discrete_move : int
        The discrete move.

    Returns
    -------
    str
        The quoridor move.
    """
    if discrete_move <= 80:
        # 0-80 are pawn moves
        # 0 -> a1
        # 1 -> b1
        # 10 -> b2
        # 79 -> h9
        # 80 -> i9
        return chr(ord("a") + discrete_move % 9) + str(discrete_move // 9 + 1)
    if discrete_move <= 144:
        # 81-144 are horizontal wall moves
        # 81 -> a1h
        # 82 -> b1h
        # 89 -> a2h
        # 90 -> b2h
        # 144 -> h8h
        return (
            chr(ord("a") + (discrete_move - 81) % 8)
            + str((discrete_move - 81) // 8 + 1)
            + "h"
        )
    if discrete_move <= 208:
        # 145-208 are vertical wall moves
        # 145 -> a1v
        # 146 -> b1v
        # 153 -> a2v
        # 154 -> b2v
        # 208 -> h8v
        return (
            chr(ord("a") + (discrete_move - 145) % 8)
            + str((discrete_move - 145) // 8 + 1)
            + "v"
        )


def convert_quoridor_move_to_discrete(move: str) -> int:
    """
    Converts a quoridor move to a discrete move Discrete(209).

    Parameters
    ----------
    move : str
        The quoridor move.

    Returns
    -------
    int
        The discrete move.
    """
    if len(move) == 2:
        # pawn move
        return (ord(move[0]) - ord("a")) + 9 * (int(move[1]) - 1)
    if move[2] == "h":
        # horizontal wall
        return 81 + (ord(move[0]) - ord("a")) + 8 * (int(move[1]) - 1)
    if move[2] == "v":
        # vertical wall
        return 145 + (ord(move[0]) - ord("a")) + 8 * (int(move[1]) - 1)


def board_to_observation(board: Quoridor):
    """
    Converts a board to an observation.

    Parameters
    ----------
    board : Dict[str,List[str]]
        The board.

    Returns
    -------
    np.ndarray
        The observation.
    """
    observation = np.zeros((9, 9, 6), dtype=bool)
    observation[*convert_cell_to_xy(board.player1.pos), 0] = 1
    observation[*convert_cell_to_xy(board.player2.pos), 1] = 1
    for wall in board.placed_walls:
        if wall[2] == "h":
            observation[*convert_cell_to_xy(wall[:2]), 2] = 1
        else:
            observation[*convert_cell_to_xy(wall[:2]), 3] = 1
    for i in range(board.player1.walls):
        observation[i // 9, i % 9, 4] = 1
    for i in range(board.player2.walls):
        observation[i // 9, i % 9, 5] = 1
    return observation


# need a number that goes like this the number 10 should be casted 10 (1, 0) and 11 (1, 1)


def convert_cell_to_xy(cell: str) -> Tuple[int, int]:
    """
    Converts a cell of the quoridor board to x and y coordinates.

    Parameters
    ----------
    cell : str
        The cell.

    Returns
    -------
    (int, int)
        The x and y coordinates.
    """
    return (
        int(cell[1]) - 1,
        ord(cell[0]) - ord("a"),
    )


# for i in range(209):
#     print(i, convet_discrete_to_quoridor_move(i))

quoridor = Quoridor.init_from_pgn("e2/e8/e3/e7/e1h")
observation = board_to_observation(quoridor)
print(observation, end="\n\n\n")
print("Layer 1: Player 1")
print(observation[:, :, 0], end="\n\n\n")
print("Layer 2: Player 2")
print(observation[:, :, 1], end="\n\n\n")
print("Layer 3: Horizontal walls")
print(observation[:, :, 2], end="\n\n\n")
print("Layer 4: Vertical walls")
print(observation[:, :, 3], end="\n\n\n")
print("Layer 5: Player 1 total_walls")
print(observation[:, :, 4], end="\n\n\n")
print("Layer 6: Player 2 total_walls")
print(observation[:, :, 5], end="\n\n\n")


# while not quoridor.is_terminated:
#     print(quoridor.current_player)
#     print(quoridor.waiting_player)
#     print(quoridor.get_legal_moves())
#     move = input("Enter move: ")
#     quoridor.make_move(move)
#     observation = board_to_observation(quoridor)
#     print(observation.shape)
#     print(observation, end="\n\n\n")
#     print("Layer 1: Player 1")
#     print(observation[:, :, 0], end="\n\n\n")
#     print("Layer 2: Player 2")
#     print(observation[:, :, 1], end="\n\n\n")
#     print("Layer 3: Horizontal walls")
#     print(observation[:, :, 2], end="\n\n\n")
#     print("Layer 4: Vertical walls")
#     print(observation[:, :, 3], end="\n\n\n")
#     print("Layer 5: Player 1 total_walls")
#     print(observation[:, :, 4], end="\n\n\n")
#     print("Layer 6: Player 2 total_walls")
#     print(observation[:, :, 5], end="\n\n\n")
print(convert_quoridor_move_to_discrete("h8v"))


action_mask = np.zeros(209, dtype=bool)
for move in quoridor.get_legal_moves():
    action_mask[convert_quoridor_move_to_discrete(move)] = True
print(action_mask[81:].sum())
