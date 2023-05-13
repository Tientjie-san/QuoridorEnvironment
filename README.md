# QuoridorEnvironment

Welcome to QuoridorEnvironment, a project aimed at creating a game environment for the board game Quoridor. This project provides an easy-to-use platform for simulating and playing the game of Quoridor, and is designed to be both flexible and customizable.

## Overview

Quoridor is a two-player board game where each player tries to reach the opposite end of the board before their opponent. The game is played on a grid of squares, and each player has a number of walls that can be placed to block the other player's progress. QuoridorEnvironment provides a way to simulate this game, and provides a framework for building agents that can play against each other. The QuordorEnv class is a subclass of PettingsZoo AECEnv

## Requirements

To use QuoridorEnvironment, you will need:

- Python 3.6 or later
- Poetry (for managing dependencies)

## Installation

To install QuoridorEnvironment, you can simply clone the repository and install the required packages:

```
git clone git@github.com:Tientjie-san/QuoridorEnvironment.git
cd QuoridorEnvironment
poetry install
```

## Usage

Once you have installed the necessary packages, you can start using QuoridorEnvironment. There are several ways to use the project:

- **Play against the built-in AI:** The project comes with a built-in AI that can play against you. To start a game against the AI, run the following command:

  ```
  poetry run python play.py
  ```

- **Create your own agent:** QuoridorEnvironment provides a framework for building your own agent that can play against other agents. To create your own agent, simply create a new Python file in the `agents` directory that defines a class that inherits from the `Agent` class in `agent.py`.
In `tournament.py` you can add your agent to the `AGENTS` variable. 
You can then run a tournament between your agent and others using the following command:

  ```
  poetry run python tournament.py
  ```

- **Use the API:** If you want to integrate QuoridorEnvironment into your own project, you can use the API provided by the `env.py` file. This file defines a `QuoridorEnvironment` class that provides methods for simulating the game and making moves. For an example on how to use the environment see `simple.py`

## Customization
Some ways you can customize the project include:

- **Create your own AI:** You can create your own AI by defining a new class that inherits from the `Agent` class in `agent.py`. You can then modify the behavior of the AI by implementing the `act` method.


## License

QuoridorEnvironment is licensed under the MIT License. See the `LICENSE` file for more details.
