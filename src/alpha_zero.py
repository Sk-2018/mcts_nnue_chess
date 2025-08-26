import math
from typing import Dict, Tuple
import chess

from state import State
from mcts import nnue  # reuse NNUE evaluation for value head

def _uniform_policy(board: chess.Board) -> Dict[str, float]:
    moves = list(board.legal_moves)
    if not moves:
        return {}
    p = 1.0 / len(moves)
    return {m.uci(): p for m in moves}

class DummyNetwork:
    """Simple placeholder network using NNUE for value and uniform policy.

    This mimics the AlphaZero interface where ``predict`` returns a policy
    distribution over legal moves and a value in [-1, 1]. The value is derived
    from the NNUE centipawn evaluation and clamped to the required range.
    """

    def predict(self, state: State) -> Tuple[Dict[str, float], float]:
        policy = _uniform_policy(state.board)
        score = nnue.nnue_evaluate_fen(state.board.fen().encode("utf-8"))
        value = max(min(score / 1000.0, 1), -1)
        return policy, value

class AZNode:
    def __init__(self, state: State, parent=None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children: Dict[str, "AZNode"] = {}
        self.N = 0  # visit count
        self.W = 0  # total value
        self.Q = 0  # mean value
        self.is_terminal = state.is_terminal()

    def expand(self, policy: Dict[str, float]):
        for action, prob in policy.items():
            if action not in self.children:
                next_state = self.state.take_action(action)
                self.children[action] = AZNode(next_state, self, prob)

    def select(self, c_puct: float):
        return max(
            self.children.items(),
            key=lambda item: item[1].Q + c_puct * item[1].prior * math.sqrt(self.N) / (1 + item[1].N),
        )

class AlphaZeroMCTS:
    def __init__(self, network: DummyNetwork, c_puct: float = 1.0, n_simulations: int = 50):
        self.network = network
        self.c_puct = c_puct
        self.n_simulations = n_simulations

    def search(self, state: State) -> str:
        self.root = AZNode(state)
        policy, value = self.network.predict(state)
        self.root.expand(policy)
        self.root.W += value
        self.root.N += 1

        for _ in range(self.n_simulations):
            node = self.root
            path = [node]
            # Selection
            while node.children:
                action, node = node.select(self.c_puct)
                path.append(node)
            # Expansion
            if not node.is_terminal:
                policy, value = self.network.predict(node.state)
                node.expand(policy)
            else:
                result = node.state.board.result()
                if result == "1-0":
                    value = 1
                elif result == "0-1":
                    value = -1
                else:
                    value = 0
            # Backup
            for n in reversed(path):
                n.N += 1
                n.W += value
                n.Q = n.W / n.N
                value = -value

        best_action, _ = max(self.root.children.items(), key=lambda item: item[1].N)
        return best_action

if __name__ == "__main__":
    searcher = AlphaZeroMCTS(DummyNetwork(), n_simulations=10)
    move = searcher.search(State())
    print("best move", move)
