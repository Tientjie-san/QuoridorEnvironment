# pylint: disable=W0613, R0903
"""Implementation of random agents"""

from quoridor import Quoridor
from Policies.policy import RandomPolicy, ShortestPathPolicy
from .agent import Agent


class RandomAgent(Agent):
    """The world's simplest agent!"""

    def __init__(self, action_space, player):
        self.action_space = action_space
        self.player: int = player
        self.random_policy = RandomPolicy()

    def act(self, observation, reward, info) -> int:
        """
        Pick a random legal action.

        Parameters
        ----------
        observation : Dict[str, np.ndarray]
            The observation.
        reward : float
            The reward.
        info : Dict[str, Any]
            The info of the environment (mainly used for debugging purposes).

        Returns
        -------
        int
            The discrete action.
        """
        return self.random_policy.get_action(observation["action_mask"])


class RandomShortestPathAgent(Agent):
    """Agent that chooses random action when player has walls and
    shortest path when player has no walls"""

    def __init__(self, action_space, player):
        self.action_space = action_space
        self.player: int = player
        self.shortest_path_policy = ShortestPathPolicy()
        self.random_policy = RandomPolicy()

    def act(self, observation, reward, info) -> int:
        """
        Pick a random legal action. If the player has no walls,
        pick the action that follows the shortest path towards the goal.

        Parameters
        ----------
        observation : Dict[str, np.ndarray]
            The observation.
        reward : float
            The reward.
        info : Dict[str, Any]
            The info of the environment (mainly used for debugging purposes).

        Returns
        -------
        int
            The discrete action.
        """
        quoridor = Quoridor.init_from_pgn(info["pgn"])
        if quoridor.current_player.walls == 0:
            return self.shortest_path_policy.get_action(quoridor)
        return self.random_policy.get_action(observation["action_mask"])
