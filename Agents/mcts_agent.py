from quoridor import Quoridor
from .agent import Agent
from Environment.utils import (
    convert_quoridor_move_to_discrete,
    convert_observation_quoridor_game,
)
from Policies.policy import ShortestPathPolicy
from math import sqrt, log
from copy import deepcopy
import random
from typing import Set

EXPLORATION_CONSTANT = 1.414


class MCTSAgent(Agent):
    """
    An agent that uses Monte Carlo Tree Search to select actions
    It utilizes the quoridor packacge to simulate games
    and the ucb1 algorithm to select actions

    state is represented as a tuple of the form:
    (player_1_pos, player_2_pos, player_1_walls, player_2_walls, placed_walls, current_player)

    How MCTS works:
    1. Tree Traversal: Selecting the best child node
    2. Expansion
    3. Simulation (Rollout): Simulating a game from the selected node until the game ends
    4. Backpropagation (Updating the tree): Updating the nodes with the results of the simulation

    Algorithm for tree traversal and expansion:
    1. Start at the root node
    2a. While the current node is fully expanded and not a leaf node:
        1. Select the best child node using the UCT formula if the current player has walls left
            or the shortest path policy if the current player has no walls left
    2b. If the current node is not fully expanded:
        1. perform rollout policy
    2c. If the current node is a leaf node and has been visited:
        1. Expand the node by adding all possible children
        2. Select a random child node
        3. Perform rollout policy

    Algorithm for rollout policy:
    1. If the game is over:
        1. Return the reward
    2. Else:
        1. Select a random action from the legal actions given the current state
        2. Perform the action
        3. overwrite the current state with the new state
        4. Repeat

    """

    class Node:
        def __init__(self, state, parent=None, action=None):
            self.state = state
            self.parent = parent
            self.children = set()
            self.total_reward = 0
            self.visits = 0
            self.action = action

        def is_terminal(self):
            """
            Returns true if the node is a terminal node
            """
            if self.state[5] == 1 and self.state[0][1] == 9:
                return True
            elif self.state[5] == 2 and self.state[1][1] == 1:
                return True
            return False

    def __init__(
        self, action_space=None, player=None, max_iterations=1000, max_time=10
    ):
        super().__init__(action_space, player)
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.root = None

    def act(self, observation, reward, info) -> int:
        """
        Selects an action using the MCTS algorithm
        """

        quoridor: Quoridor = convert_observation_quoridor_game(
            observation["observation"], self.player
        )
        # if walls are not available, use shortest path policy, to speed up the game
        if quoridor.current_player.walls == 0:
            print(quoridor.player1.walls, quoridor.player2.walls)
            print("using shortest path policy")
            return ShortestPathPolicy().get_action(quoridor)
        current_state = self._convert_quoridor_to_state(quoridor)

        if self.root is None:
            self.root = self.Node(current_state)
        action = self._search(self.root)
        return convert_quoridor_move_to_discrete(action)

    def _expand(self, node: Node):
        """
        Expands the node by adding all possible children
        """
        quoridor = self._convert_state_to_quoridor(node.state)
        for move in quoridor.get_legal_moves():
            quoridor_copy = deepcopy(quoridor)
            quoridor_copy.make_move(move)
            node.children.add(
                self.Node(
                    state=self._convert_quoridor_to_state(quoridor_copy),
                    parent=node,
                    action=move,
                )
            )

    def _rollout(self, node):
        """
        perform a rollout a from the given node
        """
        quoridor = self._convert_state_to_quoridor(node.state)
        while not quoridor.is_terminated():
            quoridor.make_move(self._rollout_policy(quoridor))

        if quoridor.current_player == self.player:
            return 1
        else:
            return -1

    def _rollout_policy(self, quoridor: Quoridor) -> str:
        """
        Selects a random action from the legal actions given the current state
        """
        return random.choice(quoridor.get_legal_moves())

    def _backpropagate(self, node: Node, reward):
        """
        Backpropagates the reward up the tree
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def _best_child(self, node):
        """
        Returns the best child of the given node
        """
        for child in node.children:
            if child.visits == 0:
                return child
        return max(node.children, key=self._uct_score)

    def _best_action(self, node: Node):
        """
        Returns the best action from the given node
        """
        return max(node.children, key=lambda x: x.visits).action

    def _search(self, root):
        """
        Searches the tree starting from the given root node
        """

        for _ in range(self.max_iterations):
            node = self._tree_traversal(root)
            reward = self._rollout(node)
            self._backpropagate(node, reward)

        return self._best_action(root)

    def _tree_traversal(self, node):
        """
        Traverses the tree starting from the given node
        """
        while not node.is_terminal():
            if len(node.children) == 0:
                self._expand(node)
                return random.choice(list(node.children.values()))
            else:
                node = self._best_child(node)
        return node

    def _uct_score(self, node: Node):
        """
        Calculates the UCT score of the given node
        """
        return node.wins / node.visits + EXPLORATION_CONSTANT * sqrt(
            log(node.parent.visits) / node.visits
        )

    def _convert_state_to_quoridor(self, state) -> Quoridor:
        """
        Converts the given state to a Quoridor game
        """
        quoridor: Quoridor = Quoridor()
        quoridor.player1.pos = state[0]
        quoridor.player2.pos = state[1]
        quoridor.player1.walls = state[2]
        quoridor.player2.walls = state[3]
        quoridor.placed_walls = state[4]
        quoridor.current_player = state[5]
        return quoridor

    def _convert_quoridor_to_state(self, quoridor: Quoridor):
        """
        Converts the given Quoridor game to a state
        """
        return (
            quoridor.player1.pos,
            quoridor.player2.pos,
            quoridor.player1.walls,
            quoridor.player2.walls,
            quoridor.placed_walls,
            quoridor.current_player,
        )
