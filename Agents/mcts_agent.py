"""
This module provides an agent implementation for Monte Carlo Tree Search (MCTS) in the Quoridor game.
"""

from typing import Optional, Set
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
import time


EXPLORATION_CONSTANT = 1.414


class MCTSAgent(Agent):
    """
    An agent that uses Monte Carlo Tree Search to select actions in the Quoridor game.
    """

    class Node:
        """
        A node in the MCTS tree.
        """

        def __init__(
            self,
            state: str,
            parent: Optional["MCTSAgent.Node"] = None,
            action: Optional[str] = None,
            terminal: bool = False,
        ) -> None:
            """
            Initializes a new Node instance.

            Parameters:
            -----------
            state: str
                The state associated with the node.
            parent: MCTSAgent.Node, optional
                The parent node of this node.
            action: str, optional
                The action that leads to this node.
            terminal: bool, optional
                Indicates whether the node represents a terminal state.
            """
            self.state: str = state
            self.parent: Optional["MCTSAgent.Node"] = parent
            self.children: Set["MCTSAgent.Node"] = set()
            self.total_reward: int = 0
            self.visits: int = 0
            self.action: Optional[str] = action
            self.is_terminal: bool = terminal

        def __str__(self) -> str:
            """
            Returns a string representation of the node.
            """
            return f"action: {self.action}, visits: {self.visits}, total_reward: {self.total_reward}"

        def __repr__(self) -> str:
            """
            Returns a string representation of the node.
            """
            return f"action: {self.action}, visits: {self.visits}, total_reward: {self.total_reward}"

    def __init__(
        self, action_space=None, player: int = None, max_iterations=10000, max_time=10
    ) -> None:
        """
        Initializes a new MCTSAgent instance.

        Parameters:
        -----------
        action_space: object, optional
            The action space.
        player: int, optional
            The player ID.
        max_iterations: int, optional
            The maximum number of iterations for the MCTS algorithm.
        max_time: int, optional
            The maximum time (in seconds) allowed for the MCTS algorithm.
        """
        super().__init__(action_space, player)
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.root: MCTSAgent.Node = None

    def act(self, observation: dict, reward: int, info: dict) -> int:
        """
        Selects an action using the MCTS algorithm.

        Parameters:
        -----------
        observation: dict
            The observation of the current state.
        reward: int
            The reward of the previous action.
        info: dict
            The info of the current state, mainly used as convenience for debugging.

        Returns:
        --------
        int:
            The action to take.
        """
        # If walls are not available, use shortest path policy to speed up the game.
        if quoridor.current_player.walls == 0:
            quoridor: Quoridor = convert_observation_quoridor_game(
                observation["observation"], self.player
            )
            return ShortestPathPolicy().get_action(quoridor)

        self.root = self.Node(info["pgn"])
        action = self._search(self.root)

        return convert_quoridor_move_to_discrete(action)

    def _get_child_node(self, node: Node, action: str) -> Optional[Node]:
        """
        Returns the child node with the given action.
        """
        for child in node.children:
            if child.action == action:
                return child

    def _expand(self, node: Node) -> None:
        """
        Expands the node by adding all possible children.
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
                    terminal=quoridor_copy.is_terminated,
                )
            )

    def _rollout(self, node: Node) -> int:
        """
        Performs a rollout from the given node.

        Parameters:
        -----------
        node: MCTSAgent.Node
            The node from which to perform the rollout.

        Returns:
        --------
        int:
            The reward obtained from the rollout.
        """
        quoridor = self._convert_state_to_quoridor(node.state)
        while not quoridor.is_terminated:
            move = self._rollout_policy(quoridor)
            quoridor.make_move(move)

        if quoridor.current_player.id == self.player:
            return 1
        else:
            return -1

    def _rollout_policy(self, quoridor: Quoridor) -> str:
        """
        Selects a random action from the legal actions given the current state.

        Parameters:
        -----------
        quoridor: Quoridor
            The Quoridor game state.

        Returns:
        --------
        str:
            The selected action.
        """
        return random.choice(list(quoridor.get_legal_pawn_moves()))

    def _backpropagate(self, node: Node, reward: int) -> None:
        """
        Backpropagates the reward up the tree.

        Parameters:
        -----------
        node: MCTSAgent.Node
            The node from which to start backpropagation.
        reward: int
            The reward to propagate.
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
            reward *= -1

    def _best_child(self, node: Node) -> Node:
        """
        Returns the best child of the given node.

        Parameters:
        -----------
        node: MCTSAgent.Node
            The node from which to select the best child.

        Returns:
        --------
        MCTSAgent.Node:
            The best child node.
        """
        for child in node.children:
            if child.visits == 0:
                return child
        return max(node.children, key=self._uct_score)

    def _best_action(self, node: Node) -> str:
        """
        Returns the best action from the given node.

        Parameters:
        -----------
        node: MCTSAgent.Node
            The node from which to select the best action.

        Returns:
        --------
        str:
            The best action.
        """
        best_node = max(node.children, key=lambda x: x.total_reward)
        return best_node.action

    def _search(self, root: Node) -> str:
        """
        Searches the tree starting from the given root node.

        Parameters:
        -----------
        root: MCTSAgent.Node
            The root node of the search.

        Returns:
        --------
        str:
            The selected action.
        """
        print("searching")
        start = time.time()

        for i in range(self.max_iterations):
            if i % 1000 == 0:
                print(i)

            node = self._tree_traversal(root)
            reward = self._rollout(node)
            self._backpropagate(node, reward)
        print(sorted(root.children, key=lambda x: x.total_reward, reverse=True)[:10])
        end = time.time()
        print(f"search time: {end-start}")
        return self._best_action(root)

    def _tree_traversal(self, node: Node) -> Node:
        """
        Traverses the tree starting from the given node.

        Parameters:
        -----------
        node: MCTSAgent.Node
            The node from which to start the traversal.

        Returns:
        --------
        MCTSAgent.Node:
            The selected node for exploration or exploitation.
        """
        while not node.is_terminal:
            if len(node.children) == 0:
                self._expand(node)
                return random.choice(list(node.children))
            else:
                node = self._best_child(node)
        return node

    def _uct_score(self, node: Node) -> float:
        """
        Calculates the UCT score of the given node.

        Parameters:
        -----------
        node: MCTSAgent.Node
            The node for which to calculate the UCT score.

        Returns:
        --------
        float:
            The UCT score of the node.
        """
        return node.total_reward / node.visits + EXPLORATION_CONSTANT * sqrt(
            log(node.parent.visits) / node.visits
        )

    def _convert_state_to_quoridor(self, state: str) -> Quoridor:
        """
        Converts the given state to a Quoridor game.

        Parameters:
        -----------
        state: str
            The state to convert.

        Returns:
        --------
        Quoridor:
            The corresponding Quoridor game.
        """
        return Quoridor.init_from_pgn(state)

    def _convert_quoridor_to_state(self, quoridor: Quoridor) -> str:
        """
        Converts the given Quoridor game to a state.

        Parameters:
        -----------
        quoridor: Quoridor
            The Quoridor game to convert.

        Returns:
        --------
        str:
            The corresponding state.
        """
        return quoridor.to_pgn()
