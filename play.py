from quoridor import Quoridor
from Environment import QuoridorEnv, env
from Agents.random_agent import RandomShortestPathAgent
from Agents.agent import Agent
from Environment.utils import convert_quoridor_move_to_discrete


class HumanAgent(Agent):
    def act(self, observation, reward, info):
        game = Quoridor.init_from_pgn(info["pgn"])
        print(f"Turn {info['turn']}:")
        print(f"Current player position: {game.current_player.pos}")
        print(f"Current opponent position: {game.waiting_player.pos}")
        print(f"Current player walls: {game.current_player.walls}")
        print(f"Current opponent walls: {game.waiting_player.walls}")
        print(f"Placed walls: {game.placed_walls}")
        print(f"Your legal moves: {game.get_legal_moves()}")
        print("Enter your move:")
        action = input()
        action = convert_quoridor_move_to_discrete(action)
        if observation["action_mask"][action] != 1:
            raise Exception("Invalid move")
        return action


quoridor_env: QuoridorEnv = env()

agents: dict[str, Agent] = {
    "player_1": HumanAgent(),
    "player_2": RandomShortestPathAgent(quoridor_env.action_spaces["player_1"], 2),
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
