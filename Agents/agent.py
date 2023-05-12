from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, action_space, player):
        self.action_space = action_space
        self.player: int = player

    @abstractmethod
    def act(self, observation, reward, info) -> int:
        """
        Pick a action.

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
        pass
