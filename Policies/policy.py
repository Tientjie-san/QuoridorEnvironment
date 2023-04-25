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
            game.board,
            game.current_player.pos,
            game.waiting_player.pos,
            game.current_player.goal,
        )[1]

        return convert_quoridor_move_to_discrete(action)

    def get_shortest_path(
        self,
        board: Dict[str, List[str]],
        current_player_pos: str,
        waiting_player_pos: str,
        goal: str,
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
        # Take into account the waiting player position for jump moves

        # Get the shortest path
        self.update_board(board, current_player_pos, waiting_player_pos)
        path = self.bfs(board, current_player_pos, goal)
        # Return the shortest path
        return path

    def update_board(
        self, board: Dict[str, List[str]], current_player_pos: str, waiting_player_pos
    ) -> None:
        """Update the nodes of the board for the current player taking into account the position of the waiting player"""
        # check if the other player is in range of current player for jumping moves
        if waiting_player_pos in board[current_player_pos]:
            board[current_player_pos].remove(waiting_player_pos)
            # same row
            if current_player_pos[1] == waiting_player_pos[1]:
                if current_player_pos[0] > waiting_player_pos[0]:
                    pos_behind = (
                        chr(ord(current_player_pos[0]) - 2) + current_player_pos[1]
                    )
                else:
                    pos_behind = (
                        chr(ord(current_player_pos[0]) + 2) + current_player_pos[1]
                    )

            elif current_player_pos[0] == waiting_player_pos[0]:  # same column
                if current_player_pos[1] > waiting_player_pos[1]:
                    pos_behind = current_player_pos[0] + chr(
                        ord(current_player_pos[1]) - 2
                    )
                else:
                    pos_behind = current_player_pos[0] + chr(
                        ord(current_player_pos[1]) + 2
                    )
            if pos_behind in board[waiting_player_pos]:
                board[current_player_pos].append(pos_behind)
            else:
                board[current_player_pos].extend(
                    pos
                    for pos in board[waiting_player_pos]
                    if pos != current_player_pos
                )

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
        action_mask : np.array
            The action mask.
        Returns
        -------
        int
            The discrete action.
        """
        return np.random.choice(np.flatnonzero(action_mask))
