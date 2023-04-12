from Environment import QuoridorEnv
from quoridor import Quoridor

quoridor = Quoridor()
quoridor.play_terminal()


from pettingzoo.classic import texas_holdem_v4

env = texas_holdem_v4.env(render_mode="human")

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample(
            observation["action_mask"]
        )  # this is where you would insert your policy
    env.step(action)
env.close()
