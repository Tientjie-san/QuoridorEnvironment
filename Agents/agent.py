from abc import ABC, abstractmethod


class Agent(ABC):
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
