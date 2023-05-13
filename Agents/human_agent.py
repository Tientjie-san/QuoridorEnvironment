from quoridor import Quoridor
from Environment.utils import convert_quoridor_move_to_discrete
from .agent import Agent


class HumanAgent(Agent):
    def __init__(self, action_space=None, player=None):
        super().__init__(action_space, player)

    def act(self, observation, reward, info):
        game = Quoridor.init_from_pgn(info["pgn"])
        print(f"Turn {info['turn']}:")
        print(f"Current player position: {game.current_player.pos}")
        print(f"Current opponent position: {game.waiting_player.pos}")
        print(f"Current player walls: {game.current_player.walls}")
        print(f"Current opponent walls: {game.waiting_player.walls}")
        print(f"Placed walls: {game.placed_walls}")
        print(f"Your legal moves: {game.get_legal_moves()}")
        print("Enter your move:")
        action = input()
        action = convert_quoridor_move_to_discrete(action)
        if observation["action_mask"][action] != 1:
            raise Exception("Invalid move")
        return action
