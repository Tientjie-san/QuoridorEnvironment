from typing import Tuple
import numpy as np
from quoridor import Quoridor


def convert_discrete_to_quoridor_move(discrete_move: int) -> str:
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
        return (ord(move[0]) - ord("a")) + 9 * (int(move[1]) - 1)
    if move[2] == "h":
        return 81 + (ord(move[0]) - ord("a")) + 8 * (int(move[1]) - 1)
    if move[2] == "v":
        return 145 + (ord(move[0]) - ord("a")) + 8 * (int(move[1]) - 1)


def board_to_observation(board: Quoridor):
    """
    Converts a board to an observation.

    Parameters
    ----------
    board : Quoridor
        The quoridor instancs.

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


def convert_xy_to_cell(x: int, y: int) -> str:
    """
    Converts x and y coordinates to a cell of the quoridor board.

    Parameters
    ----------
    x : int
        The x coordinate.
    y : int
        The y coordinate.

    Returns
    -------
    str
        The cell.
    """
    return chr(ord("a") + y) + str(x + 1)


def get_player_pos(observation: np.ndarray, player: int) -> str:
    """
    Gets the position of a player from an observation.

    Parameters
    ----------
    observation : np.ndarray
        The observation.
    player : int
        The player.

    Returns
    -------
    str
        The position.
    """
    x, y = np.where(observation[:, :, player - 1])
    return convert_xy_to_cell(x[0], y[0])


def convert_observation_quoridor_game(observation: np.ndarray, player: int) -> Quoridor:
    """
    Converts an observation to a quoridor game.

    Parameters
    ----------
    observation : np.ndarray
        The observation.
    player : int
        The current player.

    Returns
    -------
    Quoridor
        The quoridor game.
    """

    quoridor = Quoridor()
    if player == 1:
        quoridor.current_player = quoridor.player1
        quoridor.waiting_player = quoridor.player2
    else:
        quoridor.current_player = quoridor.player2
        quoridor.waiting_player = quoridor.player1

    quoridor.current_player.pos = get_player_pos(observation, player)
    quoridor.waiting_player.pos = get_player_pos(observation, 3 - player)
    quoridor.current_player.walls = np.sum(observation[:, :, 4 - player])
    quoridor.waiting_player.walls = np.sum(observation[:, :, 5 - player])
    quoridor.placed_walls = []
    for channel in range(2, 4):
        x, y = np.where(observation[:, :, channel])
        for i in range(len(x)):
            if channel == 2:
                quoridor.placed_walls.append(convert_xy_to_cell(x[i], y[i]) + "h")
            else:
                quoridor.placed_walls.append(convert_xy_to_cell(x[i], y[i]) + "v")
    for wall in quoridor.placed_walls:
        quoridor._remove_connections(quoridor.board, wall)
    return quoridor
