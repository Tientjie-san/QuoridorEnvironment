from Environment import QuoridorEnv, env
from Vizualiser.vizualiser import QuoridorVizualiser
from Agents.random_agent import RandomAgent

quoridor_env: QuoridorEnv = env()
quoridor_vizualiser = QuoridorVizualiser()

quoridor_env.reset()

agents = {
    "player_1": RandomAgent(quoridor_env.action_spaces["player_1"], "player_1"),
    "player_2": RandomAgent(quoridor_env.action_spaces["player_2"], "player_2"),
}

for agent in quoridor_env.agent_iter():
    observation, reward, termination, truncation, info = quoridor_env.last()
    if termination:
        break
    action = RandomPolicy().get_action(observation["action_mask"])
    quoridor_env.step(action)

quoridor_vizualiser.render_game(info["pgn"])


# from pettingzoo.test import api_test  # noqa: E402

# if __name__ == "__main__":
#     api_test(QuoridorEnv(), num_cycles=1_000_000)
