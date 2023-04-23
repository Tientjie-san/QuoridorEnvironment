"""Vizualiser is a Python package for visualizing quoridor games"""

from quoridor import Quoridor
import pygame


class QuoridorVizualiser:
    def render(self, pgn: str, mode: str = "human"):
        """Render the latest position of the game

        Parameters
        ----------
        pgn : str
            The PGN of the Quoridor game.
        mode : str, optional
            The mode, by default "human".
        """
        if mode == "human":
            self._render_human(pgn)
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented")

    def render_game(self, pgn: str, mode: str = "human"):
        """Render the game

        Parameters
        ----------
        game : Quoridor
            The game.
        mode : str, optional
            The mode, by default "human".
        """

        self.render(pgn, mode)

    def _render_human(self, pgn: str):
        """Render the game in human mode

        Parameters
        ----------
        game : Quoridor
            The game.
        """
        pass
