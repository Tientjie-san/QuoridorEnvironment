from quoridor import Quoridor
from .agent import Agent
from Environment.utils import (
    convert_quoridor_move_to_discrete,
    convert_observation_quoridor_game,
)


class MCTSAgent(Agent):
    """
    An agent that uses Monte Carlo Tree Search to select actions
    It utilizes the quoridor packacge to simulate games
    and the ucb1 algorithm to select actions

    state is represented as a tuple of the form:
    (player_1_pos, player_2_pos, player_1_walls, player_2_walls, placed_walls)
    """

    class Node:
        def __init__(self, state, parent=None):
            self.state = state
            self.parent = parent
            self.children = []
            self.wins = 0
            self.visits = 0

    def __init__(
        self, action_space=None, player=None, max_iterations=1000, max_time=10
    ):
        super().__init__(action_space, player)
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.root = None

    def act(self, observation, reward, info):
        """
        Selects an action using the MCTS algorithm
        """
        current_state = convert_observation_quoridor_game(observation, self.player)
        if self.root is None:
            self.root = self.Node(current_state)

    def _expand(self, node):
        """
        Expands the node by adding all possible children
        """
        pass

    def _simulate(self, node):
        """
        Simulates a game from the given node
        """
        pass

    def _backpropagate(self, node, reward):
        """
        Backpropagates the reward up the tree
        """
        pass

    def _best_child(self, node):
        """
        Returns the best child of the given node
        """
        pass

    def _best_action(self, node):
        """
        Returns the best action from the given node
        """
        pass

    def _search(self, root):
        """
        Searches the tree starting from the given root node
        """
        pass

    def _uct_score(self, node):
        """
        Calculates the UCT score of the given node
        """
        pass
