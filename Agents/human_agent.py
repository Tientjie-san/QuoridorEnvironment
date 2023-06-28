"""
This module provides an agent implementation for human players in the Quoridor game.
"""

from quoridor import Quoridor
from Environment.utils import convert_quoridor_move_to_discrete
from .agent import Agent


class HumanAgent(Agent):
    """
    Agent implementation for human players.
    """

    def __init__(self, action_space=None, player=None):
        """
        Initialize the HumanAgent.

        Parameters
        ----------
        action_space : object, optional
            The action space of the environment.
        player : object, optional
            The player identifier.
        """
        super().__init__(action_space, player)

    def act(self, observation, reward, info):
        """
        Allows a human player to input their move.

        Parameters
        ----------
        observation : dict
            The observation from the environment.
        reward : float
            The reward from the previous action.
        info : dict
            Additional information about the game state.

        Returns
        -------
        int
            The selected action.
        """
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
