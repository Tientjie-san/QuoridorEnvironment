from Environment import QuoridorEnv, env
from Agents import RandomShortestPathAgent, HumanAgent, MCTSAgent
from Agents.agent import Agent


def play():
    quoridor_env: QuoridorEnv = env()
    agents: dict[str, Agent] = {
        "player_1": HumanAgent(),
        "player_2": MCTSAgent(quoridor_env.action_spaces["player_2"], 2),
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


print("Hello World")
if __name__ == "__main__":
    play()
