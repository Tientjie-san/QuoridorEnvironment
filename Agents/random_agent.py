from Policies.policy import RandomPolicy, ShortestPathPolicy
from quoridor import Quoridor


class RandomAgent:
    """The world's simplest agent!"""

    def __init__(self, action_space, player):
        self.action_space = action_space
        self.player: int = player
        self.shortest_path_policy = ShortestPathPolicy()
        self.random_policy = RandomPolicy()

    def act(self, observation, reward, info):
        # quoridor = Quoridor.init_from_pgn(info["pgn"])
        # if quoridor.current_player.walls == 0:
        #     action = self.shortest_path_policy.get_action(quoridor)
        #     if observation["action_mask"][action] == False:
        #         print(action)
        #         raise Exception("Invalid action")
        # else:
        return self.random_policy.get_action(observation["action_mask"])
