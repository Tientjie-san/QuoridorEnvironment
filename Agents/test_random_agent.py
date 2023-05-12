# pylint: skip-file
from .random_agent import RandomAgent, RandomShortestPathAgent
from Environment import QuoridorEnv, env


def test_random_agent():
    quoridor_env: QuoridorEnv = env()
    quoridor_env.reset()
    observation, reward, termination, truncation, info = quoridor_env.last()
    random_agent = RandomAgent(None, 1)
    legal_action_mask = observation["action_mask"]
    assert legal_action_mask[random_agent.act(observation, reward, info)] == 1


def test_random_shortest_path_agent():
    quoridor_env: QuoridorEnv = env()
    agents = {
        "player_1": RandomShortestPathAgent(quoridor_env.action_spaces["player_1"], 1),
        "player_2": RandomShortestPathAgent(quoridor_env.action_spaces["player_2"], 2),
    }
    quoridor_env.reset()
    for agent in quoridor_env.agent_iter():
        observation, reward, termination, truncation, info = quoridor_env.last()
        if termination:
            break

        action = agents[agent].act(observation, reward, info)
        legal_action_mask = observation["action_mask"]
        assert legal_action_mask[action] == 1

        quoridor_env.step(action)
