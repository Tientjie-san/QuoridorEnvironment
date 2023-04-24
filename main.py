from Environment import QuoridorEnv, env
from Vizualiser.vizualiser import QuoridorVizualiser
from Agents.random_agent import RandomAgent
from Policies.policy import RandomPolicy
import time

quoridor_env: QuoridorEnv = env()
quoridor_vizualiser = QuoridorVizualiser()
EPISODES = 100

agents = {
    "player_1": RandomAgent(quoridor_env.action_spaces["player_1"], "player_1"),
    "player_2": RandomAgent(quoridor_env.action_spaces["player_2"], "player_2"),
}
start_time = time.time()
wins = 0
total_turns = 0

for i in range(EPISODES):
    print(f"Episode {i+1}")
    quoridor_env.reset()
    for agent in quoridor_env.agent_iter():
        observation, reward, termination, truncation, info = quoridor_env.last()
        if termination:
            if quoridor_env.rewards["player_1"] == 1:
                wins += 1
                total_turns += info["turn"]
            break
        action = agents[agent].act(observation, reward, info)
        quoridor_env.step(action)


end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time} seconds")
print(f"win rate: {int(wins / EPISODES *100)}% ")
print(f"average turns: {total_turns / EPISODES}")
# quoridor_vizualiser.render_game(info["pgn"])
# print(quoridor_env.rewards)
# print(quoridor_env.infos)
# print(type(observation["observation"]))

from pettingzoo.test import api_test  # noqa: E402

# if __name__ == "__main__":
#     api_test(QuoridorEnv(), num_cycles=1_000_000)
