# pylint: skip-file
from quoridor import Quoridor
import numpy as np
from .utils import (
    convert_discrete_to_quoridor_move,
    convert_quoridor_move_to_discrete,
    board_to_observation,
    convert_observation_quoridor_game,
)
from .env import QuoridorEnv, env


def test_convert_discrete_to_quoridor_move():
    # test range of discrete moves to quoridor moves test pawn moves and wall moves

    assert convert_discrete_to_quoridor_move(0) == "a1"
    assert convert_discrete_to_quoridor_move(1) == "b1"
    assert convert_discrete_to_quoridor_move(10) == "b2"
    assert convert_discrete_to_quoridor_move(79) == "h9"
    assert convert_discrete_to_quoridor_move(80) == "i9"
    assert convert_discrete_to_quoridor_move(81) == "a1h"
    assert convert_discrete_to_quoridor_move(144) == "h8h"
    assert convert_discrete_to_quoridor_move(145) == "a1v"
    assert convert_discrete_to_quoridor_move(208) == "h8v"


def test_convert_quoridor_move_to_discrete():
    assert convert_quoridor_move_to_discrete("a1") == 0
    assert convert_quoridor_move_to_discrete("i9") == 80
    assert convert_quoridor_move_to_discrete("a1h") == 81
    assert convert_quoridor_move_to_discrete("h8h") == 144
    assert convert_quoridor_move_to_discrete("a1v") == 145
    assert convert_quoridor_move_to_discrete("h8v") == 208


def test_board_to_observation():
    quoridor = Quoridor.init_from_pgn("e2/e8/e3/e7/e1h/e3v/e2h")
    observation = board_to_observation(quoridor)
    # assert if player 1 position is correct
    assert observation[:, :, 0][2, 4] == 1
    # assert if player 2 position is correct
    assert observation[:, :, 1][6, 4] == 1
    # assert if horizontal walls are correct
    assert observation[:, :, 2][0, 4] == 1
    # assert if vertical walls are correct
    assert observation[:, :, 3][2, 4] == 1
    # assert if player 1 total walls are correct)
    assert np.sum(observation[:, :, 4]) == 8
    # assert if player 2 total walls are correct
    assert np.sum(observation[:, :, 5]) == 9


def test_convert_observation_quoridor_game():
    quoridor = Quoridor.init_from_pgn("e2/e8/e3/e7/e1h/e3v/e2h")
    observation = board_to_observation(quoridor)
    print("layer 1: Player 1 position:\n", observation[:, :, 0])
    print("layer 2: Player 2 position\n", observation[:, :, 1])
    print("layer 3: Player 1 placed walls \n", observation[:, :, 2])
    print("layer 4: Player 2 placed walls \n", observation[:, :, 3])
    print("layer 5: Player 1 total walls\n", observation[:, :, 4])
    print("layer6: Player 2 total walls\n", observation[:, :, 5])

    board = convert_observation_quoridor_game(observation, 1)
    assert board.current_player.id == 1
    assert set(board.placed_walls) == set(["e1h", "e2h", "e3v"])
    assert board.player1.pos == "e3"
    assert board.player2.pos == "e7"
    assert board.player1.walls == 8
    assert board.player2.walls == 9
    assert board.current_player.walls == 8
    assert board.waiting_player.walls == 9

    board = convert_observation_quoridor_game(observation, 2)
    assert board.current_player.id == 2
    assert set(board.placed_walls) == set(["e1h", "e2h", "e3v"])
    assert board.player1.pos == "e3"
    assert board.player2.pos == "e7"
    assert board.player1.walls == 8
    assert board.player2.walls == 9
    assert board.current_player.walls == 9
    assert board.waiting_player.walls == 8


def test_env():
    quoridor_env: QuoridorEnv = env()
