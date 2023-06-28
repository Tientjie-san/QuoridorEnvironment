"""
This module provides a function for playing a game of Quoridor.
"""

from Environment import QuoridorEnv, env
from Agents import *
from Agents.agent import Agent


def play():
    """
    Play a game of Quoridor.

    The game is played between a human player and a shortest path agent.

    The human player is controlled through the terminal, while the shortest path agent
    uses a predetermined algorithm to make its moves.
    """
    quoridor_env: QuoridorEnv = env()
    agents: dict[str, Agent] = {
        "player_1": HumanAgent(),
        "player_2": ShortestPathAgent(quoridor_env.action_spaces["player_2"], 2),
    }

    quoridor_env.reset()
    for agent in quoridor_env.agent_iter():
        observation, reward, termination, truncation, info = quoridor_env.last()
        if termination:
            if quoridor_env.rewards["player_1"] == 1:
                print("You won!")
            else:
                print("You lost!")
            break
        action = agents[agent].act(observation, reward, info)
        quoridor_env.step(action)


if __name__ == "__main__":
    play()
