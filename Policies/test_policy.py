# pylint: skip-file
from quoridor import Quoridor
from .policy import ShortestPathPolicy


def test_get_shortest_path():
    quoridor = Quoridor()
    policy = ShortestPathPolicy()
    shorest_path = policy.get_shortest_path(
        quoridor.board,
        quoridor.current_player.pos,
        quoridor.waiting_player.pos,
        quoridor.current_player.goal,
    )
    assert shorest_path[1:] == ["e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"]

    quoridor = Quoridor()
    quoridor.current_player.pos = "e4"
    quoridor.waiting_player.pos = "e5"
    shorest_path = policy.get_shortest_path(
        quoridor.board,
        quoridor.current_player.pos,
        quoridor.waiting_player.pos,
        quoridor.current_player.goal,
    )
    assert shorest_path[1:] == ["e6", "e7", "e8", "e9"]
