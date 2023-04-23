from Environment import QuoridorEnv, env
from Policies.policy import ShortestPathPolicy, RandomPolicy
from Vizualiser.vizualiser import QuoridorVizualiser

quoridor_env: QuoridorEnv = env()
quoridor_vizualiser = QuoridorVizualiser()

quoridor_env.reset()

for agent in quoridor_env.agent_iter():
    observation, reward, termination, truncation, info = quoridor_env.last()
    # print(info)
    if termination:
        break
    action = RandomPolicy().get_action(observation["action_mask"])
    quoridor_env.step(action)

quoridor_vizualiser.render_game(info["pgn"])

# # # print(quoridor_env.observe("player_1"))
# # observation = quoridor_env.observe("player_1")["observation"]
# # print("Layer 1: Player 1")
# # print(observation[:, :, 0], end="\n\n\n")
# # print("Layer 2: Player 2")
# # print(observation[:, :, 1], end="\n\n\n")
# # print("Layer 3: Horizontal walls")
# # print(observation[:, :, 2], end="\n\n\n")
# # print("Layer 4: Vertical walls")
# # print(observation[:, :, 3], end="\n\n\n")
# # print("Layer 5: Player 1 total_walls")
# # print(observation[:, :, 4], end="\n\n\n")
# # print("Layer 6: Player 2 total_walls")
# # print(observation[:, :, 5], end="\n\n\n")
# # print("Action mask")
# # print(quoridor_env.observe("player_1")["action_mask"], end="\n\n\n")
# # quoridor_env = QuoridorEnv()
# # quoridor_env.reset()
# # quoridor_env.last()
# print(action_mask)
# action_mask_indices = np.flatnonzero(action_mask)
# print(action_mask_indices)
# print(random.choice(list(action_mask_indices)))
# policy = ShortestPathPolicy()
# quoridor = Quoridor.init_from_pgn("e2/e8/e3/e7/e4")
# print(quoridor.current_player)
# print(
#     ShortestPathPolicy().get_shortest_path(
#         quoridor.board, quoridor.current_player.pos, quoridor.current_player.goal
#     )
# )
# move = ShortestPathPolicy().get_action(quoridor)
# print(move, convet_discrete_to_quoridor_move(move))
# move = RandomPolicy().get_action(
#     quoridor_env.action_space(quoridor_env.agent_iter()), action_mask
# )
# print(move, convet_discrete_to_quoridor_move(move))

# for agent in quoridor_env.agent_iter():
#     observation, reward, termination, truncation, info = quoridor_env.last()
#     action = policy(observation, agent)
#     quoridor_env.step(action)


# from pettingzoo.test import api_test  # noqa: E402

# if __name__ == "__main__":
#     api_test(QuoridorEnv(), num_cycles=1_000_000)
