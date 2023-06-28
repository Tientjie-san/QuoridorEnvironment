"""
This module provides an abstract base class for defining agents in a reinforcement learning environment.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class Agent(ABC):
    """
    Agent base class.
    """

    def __init__(self, action_space, player):
        """
        Initialize the Agent.

        Parameters
        ----------
        action_space : object
            The action space of the environment.
        player : int
            The player identifier.
        """
        self.action_space = action_space
        self.player: int = player

    @abstractmethod
    def act(
        self, observation: Dict[str, np.ndarray], reward: float, info: Dict[str, Any]
    ) -> int:
        """
        Pick an action.

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
