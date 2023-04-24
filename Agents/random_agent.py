from Policies.policy import RandomPolicy, ShortestPathPolicy
from quoridor import Quoridor


class RandomAgent:
    """The world's simplest agent!"""

    def __init__(self, action_space, player):
        self.action_space = action_space
        self.player = player
        self.shortest_path_policy = ShortestPathPolicy()
        self.random_policy = RandomPolicy()

    def act(self, observation, reward):
        walls = observation["observation"][:, :, 2:4]
        if walls == 0:
            return self.shortest_path_policy.get_action(observation["action_mask"])
        else:
            return self.random_policy.get_action(observation["action_mask"])

    def convert_observation_to_game(self, observation):
        """Convert the observation to a quoridor game"""
        game = Quoridor()
        game.player1.pos = ...
        game.player2.pos = ...
        game.player1.goal = ...
        board.board = observation["observation"][:, :, 0:2]
        return board
