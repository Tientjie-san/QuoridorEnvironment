from Agents.mcts_agent import MCTSAgent
from Environment import QuoridorEnv, env
from Environment.utils import (
    convert_quoridor_move_to_discrete,
    convert_observation_quoridor_game,
)
from quoridor import Quoridor


def test_node_initialization():
    state = "e2/e8"
    parent = MCTSAgent.Node(state="e2")
    action = "e8"
    terminal = False
    node = MCTSAgent.Node(state, parent, action, terminal)

    assert node.state == state
    assert node.parent == parent
    assert node.children == set()
    assert node.total_reward == 0
    assert node.visits == 0
    assert node.action == action
    assert node.is_terminal == terminal


def test_expand():
    mcts_agent: MCTSAgent = MCTSAgent(player=1, max_iterations=5)
    quoridor_game: Quoridor = Quoridor()

    state = quoridor_game.get_pgn()
    node = MCTSAgent.Node(state)

    assert len(node.children) == 0

    mcts_agent._expand(node)

    assert len(node.children) > 0
    for child in node.children:
        assert isinstance(child, MCTSAgent.Node)
        assert child.parent == node
        assert child.action in quoridor_game.get_legal_moves()
        assert child.is_terminal == False


def test_rollout():
    mcts_agent: MCTSAgent = MCTSAgent(player=1, max_iterations=5)
    quoridor_game: Quoridor = Quoridor()
    state = quoridor_game.get_pgn()
    node = MCTSAgent.Node(state)

    reward = mcts_agent._rollout(node)

    assert reward in [-1, 1]


def test_rollout_policy():
    mcts_agent: MCTSAgent = MCTSAgent(player=1, max_iterations=5)
    quoridor_game: Quoridor = Quoridor()
    legal_moves = quoridor_game.get_legal_pawn_moves()

    action = mcts_agent._rollout_policy(quoridor_game)

    assert action in legal_moves


def test_backpropagate():
    mcts_agent: MCTSAgent = MCTSAgent(player=1, max_iterations=5)
    quoridor_game: Quoridor = Quoridor()
    root = MCTSAgent.Node(state=quoridor_game.get_pgn())
    child1 = MCTSAgent.Node(state="", parent=root)
    child2 = MCTSAgent.Node(state="", parent=root)
    grandchild1 = MCTSAgent.Node(state="", parent=child1)
    grandchild2 = MCTSAgent.Node(state="", parent=child1)

    reward = 1

    mcts_agent._backpropagate(grandchild1, reward)

    assert grandchild1.visits == 1
    assert grandchild1.total_reward == 1
    assert grandchild2.visits == 0
    assert grandchild2.total_reward == 0
    assert child1.visits == 1
    assert child1.total_reward == -1
    assert child2.visits == 0
    assert child2.total_reward == 0


def test_act():
    mcts_agent = MCTSAgent(player=1, max_iterations=5)
    quordior_env: QuoridorEnv = env()
    quordior_env.reset()
    observation, reward, termination, truncation, info = quordior_env.last()

    action = mcts_agent.act(observation, reward, info)
    assert isinstance(action, int)
    assert observation["action_mask"][action] == 1
