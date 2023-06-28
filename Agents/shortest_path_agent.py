from quoridor import Quoridor
from .agent import Agent
from Policies.policy import ShortestPathPolicy


class ShortestPathAgent(Agent):
    def __init__(self, action_space, player):
        """
        Initializes a ShortestPathAgent.

        Parameters:
        -----------
        action_space: object
            The action space of the environment.
        player: int
            The player ID associated with the agent.
        """
        super().__init__(action_space, player)
        self.shortest_path_policy = ShortestPathPolicy()

    def act(self, observation, reward, info) -> int:
        """
        Selects an action using the shortest path policy.

        Parameters:
        -----------
        observation: dict
            The observation from the environment.
        reward: float
            The reward from the previous action.
        info: dict
            Additional information about the game state.

        Returns:
        --------
        int:
            The selected action.
        """
        quoridor = Quoridor.init_from_pgn(info["pgn"])
        return self.shortest_path_policy.get_action(quoridor)
