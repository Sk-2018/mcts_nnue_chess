import math
from typing import Dict, Tuple
import chess
import torch

from state import State


def _uniform_policy(board: chess.Board) -> Dict[str, float]:
    moves = list(board.legal_moves)
    if not moves:
        return {}
    p = 1.0 / len(moves)
    return {m.uci(): p for m in moves}


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode a board into a flat 768-dimensional tensor."""
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)
    for square, piece in board.piece_map().items():
        idx = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
        tensor[idx, square // 8, square % 8] = 1.0
    return tensor.view(-1)


class TinyNetwork(torch.nn.Module):
    """Minimal trainable value network used by AlphaZeroMCTS."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(12 * 64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, state: State) -> Tuple[Dict[str, float], float]:
        policy = _uniform_policy(state.board)
        x = board_to_tensor(state.board).unsqueeze(0)
        value = self.forward(x).item()
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
    def __init__(self, network: TinyNetwork, c_puct: float = 1.0, n_simulations: int = 50):
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


class AlphaZeroTrainer:
    """Very small self-play reinforcement learning loop."""

    def __init__(self, network: TinyNetwork, lr: float = 1e-3, simulations: int = 25):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.searcher = AlphaZeroMCTS(network, n_simulations=simulations)
        self.loss_fn = torch.nn.MSELoss()

    def self_play(self):
        state = State()
        tensors = []
        players = []
        while not state.is_terminal():
            tensors.append(board_to_tensor(state.board))
            players.append(1 if state.board.turn == chess.WHITE else -1)
            move = self.searcher.search(state)
            state = state.take_action(move)
        result = state.board.result()
        if result == "1-0":
            z = 1
        elif result == "0-1":
            z = -1
        else:
            z = 0
        targets = torch.tensor([z * p for p in players], dtype=torch.float32)
        inputs = torch.stack(tensors)
        return inputs, targets

    def train_step(self) -> float:
        inputs, targets = self.self_play()
        self.optimizer.zero_grad()
        outputs = self.network(inputs).squeeze()
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    net = TinyNetwork()
    trainer = AlphaZeroTrainer(net, simulations=5)
    loss = trainer.train_step()
    print("training loss", loss)
