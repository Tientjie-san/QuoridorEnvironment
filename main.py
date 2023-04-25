from Environment import QuoridorEnv, env
from Vizualiser.vizualiser import QuoridorVizualiser
from Agents.random_agent import RandomAgent, RandomShortestPathAgent
import time

quoridor_env: QuoridorEnv = env()
quoridor_vizualiser = QuoridorVizualiser()
EPISODES = 100

agents = {
    "player_1": RandomShortestPathAgent(quoridor_env.action_spaces["player_1"], 1),
    "player_2": RandomAgent(quoridor_env.action_spaces["player_2"], 2),
}
start_time = time.time()
wins = 0
total_turns = 0
games = []

for i in range(EPISODES):
    print(f"Episode {i+1}")
    quoridor_env.reset()
    for agent in quoridor_env.agent_iter():
        observation, reward, termination, truncation, info = quoridor_env.last()
        if termination:
            if quoridor_env.rewards["player_1"] == 1:
                wins += 1

            total_turns += info["turn"]
            games.append(info["pgn"])
            break

        action = agents[agent].act(observation, reward, info)
        if action == None:
            raise Exception("action is None")
        quoridor_env.step(action)


end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")
print(f"win rate: {(wins / EPISODES *100):.2f}% ")
print(f"average turns: {(total_turns / EPISODES):.2f}")

# TODO Store games to your data store

# from pettingzoo.test import api_test  # noqa: E402

# if __name__ == "__main__":
#     api_test(QuoridorEnv(), num_cycles=1_000_000)
