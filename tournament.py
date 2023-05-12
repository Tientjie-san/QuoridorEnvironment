"""
A tournament is a round robin competition where every agent plays every other agent twice (once as player 1 and once as player 2)
The ranking is based on the total number of wins

"""
from collections import Counter
from Agents import RandomAgent, RandomShortestPathAgent
from Agents.agent import Agent
from Environment import QuoridorEnv, env

AGENTS = [RandomAgent, RandomShortestPathAgent]


class Tournament:
    def __init__(self, agents: list[Agent]):
        self.agents = agents
        self.ranking = Counter({agent: 0 for agent in agents})

    def run(self):
        # every agent plays every other agent twice
        for agent_1 in self.agents:
            for agent_2 in self.agents:
                if agent_1 == agent_2:
                    continue
                self.play(agent_1, agent_2)

    def play(self, agent_1: Agent, agent_2: Agent):
        quoridor_env: QuoridorEnv = env()

        agents = {
            "player_1": agent_1(quoridor_env.action_spaces["player_1"], 1),
            "player_2": agent_2(quoridor_env.action_spaces["player_2"], 2),
        }

        quoridor_env.reset()
        for agent in quoridor_env.agent_iter():
            observation, reward, termination, truncation, info = quoridor_env.last()
            if termination:
                if quoridor_env.rewards["player_1"] == 1:
                    self.ranking[agent_1] += 1
                else:
                    self.ranking[agent_2] += 1

                break
            action = agents[agent].act(observation, reward, info)
            quoridor_env.step(action)

    def get_ranking(self):
        return self.ranking.most_common()


if __name__ == "__main__":
    tournament = Tournament(AGENTS)
    tournament.run()
    print(tournament.ranking)
