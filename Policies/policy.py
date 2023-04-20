from typing import Dict, List
from quoridor import Quoridor
from Environment.utils import convert_quoridor_move_to_discrete
import numpy as np


class ShortestPathPolicy:
    """
    Shortest path policy
    This policy is used to find the action that follows the shortest path between two nodes
    """

    def get_action(self, game: Quoridor) -> int:
        """
        Get the action that follows the shortest path between two nodes

        Parameters
        ----------
        board : Quoridor
            The board.
        Returns
        -------
        int
            The discrete action.
        """

        action = self.get_shortest_path(
            game.board, game.current_player.pos, game.current_player.goal
        )[1]

        return convert_quoridor_move_to_discrete(action)

    def get_shortest_path(
        self, board: Dict[str, List[str]], pos: str, goal: str
    ) -> List[str]:
        """Get the shortest path between the player position and the goal
        Parameters
        ----------
        board : Dict[str, List[str]]
            The board.
        pos : str
            The player position.
        goal : str
            The goal.
        Returns
        -------
        List[str]
            The shortest path.
        """

        # Get the shortest path
        path = self.bfs(board, pos, goal)
        # Return the shortest path
        return path

    def bfs(self, board: Dict[str, List[str]], pos: str, goal: str) -> List[str]:
        """Breadth-first search
        Parameters
        ----------
        board : Dict[str, List[str]]
            The board.
        pos : str
            The player position.
        goal : str
            The goal.
        Returns
        -------
        List[str]
            The shortest path.
        """

        # Initialize the queue
        queue = [[pos]]
        # Initialize the visited nodes
        visited = set()
        # While the queue is not empty
        while queue:
            # Get the path
            path = queue.pop(0)
            # Get the node
            node = path[-1]
            # If the node is not visited
            if node not in visited:
                # Get the neighbors
                neighbors = board[node]
                # For each neighbor
                for neighbor in neighbors:
                    # Get the new path
                    new_path = list(path)
                    # Append the neighbor to the new path
                    new_path.append(neighbor)
                    # Add the new path to the queue
                    queue.append(new_path)
                    # If the neighbor is the goal
                    if neighbor[1] == goal:
                        # Return the new path
                        return new_path
                # Add the node to the visited nodes
                visited.add(node)
        # Return an empty list
        return []


class RandomPolicy:
    """
    Random policy
    This policy is used to find a random action
    """

    def get_action(self, action_mask) -> int:
        """Get a random action
        ----------
        observation : np.array
            The observation.
        action_mask : np.array
            The action mask.
        Returns
        -------
        int
            The discrete action.
        """

        # Get the legal actions
        legal_actions = [i for i in range(209) if action_mask[i]]
        # Get a random action
        action = np.random.choice(legal_actions)
        # Return the action
        return action
