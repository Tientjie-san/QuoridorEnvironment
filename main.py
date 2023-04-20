from Environment import QuoridorEnv, env
from quoridor import Quoridor

#
quoridor_env = env()
quoridor_env.reset()
# print(quoridor_env.observe("player_1"))
observation = quoridor_env.observe("player_1")["observation"]
print("Layer 1: Player 1")
print(observation[:, :, 0], end="\n\n\n")
print("Layer 2: Player 2")
print(observation[:, :, 1], end="\n\n\n")
print("Layer 3: Horizontal walls")
print(observation[:, :, 2], end="\n\n\n")
print("Layer 4: Vertical walls")
print(observation[:, :, 3], end="\n\n\n")
print("Layer 5: Player 1 total_walls")
print(observation[:, :, 4], end="\n\n\n")
print("Layer 6: Player 2 total_walls")
print(observation[:, :, 5], end="\n\n\n")
print("Action mask")
print(quoridor_env.observe("player_1")["action_mask"], end="\n\n\n")
# quoridor_env = QuoridorEnv()
# quoridor_env.reset()
# quoridor_env.last()
