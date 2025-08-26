import math
from typing import Dict, Tuple
import chess
import torch

from state import State


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode a board including side to move and castling rights."""
    tensor = torch.zeros(17, 8, 8, dtype=torch.float32)
    for square, piece in board.piece_map().items():
        idx = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
        tensor[idx, square // 8, square % 8] = 1.0
    # Side to move plane
    tensor[12].fill_(1.0 if board.turn == chess.WHITE else 0.0)
    # Castling rights
    tensor[13].fill_(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    tensor[14].fill_(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    tensor[15].fill_(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    tensor[16].fill_(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    return tensor.view(-1)


def move_to_index(move: chess.Move) -> int:
    """Map a move to an index in [0, 4095] ignoring promotions."""
    return move.from_square * 64 + move.to_square


class TinyNetwork(torch.nn.Module):
    """Minimal policy/value network used by AlphaZeroMCTS."""

    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(17 * 64, 128),
            torch.nn.ReLU(),
        )
        self.policy_head = torch.nn.Linear(128, 64 * 64)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.policy_head(h), self.value_head(h)

    def predict(self, state: State) -> Tuple[Dict[str, float], float]:
        x = board_to_tensor(state.board).unsqueeze(0)
        logits, value = self.forward(x)
        probs = torch.softmax(logits[0], dim=0)
        policy: Dict[str, float] = {}
        for move in state.board.legal_moves:
            idx = move_to_index(move)
            policy[move.uci()] = probs[idx].item()
        total = sum(policy.values())
        if total > 0:
            policy = {m: p / total for m, p in policy.items()}
        else:
            # fall back to uniform if logits underflow
            p = 1.0 / len(policy) if policy else 0.0
            policy = {m: p for m in policy}
        return policy, value.item()

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

    def search(self, state: State) -> Tuple[str, Dict[str, float]]:
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
        total = sum(child.N for child in self.root.children.values())
        pi = {a: child.N / total for a, child in self.root.children.items()} if total > 0 else {}
        return best_action, pi


class AlphaZeroTrainer:
    """Very small self-play reinforcement learning loop."""

    def __init__(self, network: TinyNetwork, lr: float = 1e-3, simulations: int = 25):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.searcher = AlphaZeroMCTS(network, n_simulations=simulations)
        self.value_loss = torch.nn.MSELoss()

    def self_play(self):
        state = State()
        tensors = []
        players = []
        policies = []
        while not state.is_terminal():
            tensors.append(board_to_tensor(state.board))
            players.append(1 if state.board.turn == chess.WHITE else -1)
            move, pi = self.searcher.search(state)
            policy_vec = torch.zeros(64 * 64)
            for m, p in pi.items():
                policy_vec[move_to_index(chess.Move.from_uci(m))] = p
            policies.append(policy_vec)
            state = state.take_action(move)
        result = state.board.result()
        if result == "1-0":
            z = 1
        elif result == "0-1":
            z = -1
        else:
            z = 0
        value_targets = torch.tensor([z * p for p in players], dtype=torch.float32)
        inputs = torch.stack(tensors)
        policy_targets = torch.stack(policies)
        return inputs, policy_targets, value_targets

    def train_step(self) -> float:
        inputs, policy_t, value_t = self.self_play()
        self.optimizer.zero_grad()
        policy_logits, values = self.network(inputs)
        v_loss = self.value_loss(values.squeeze(), value_t)
        log_probs = torch.log_softmax(policy_logits, dim=1)
        p_loss = -(policy_t * log_probs).sum(dim=1).mean()
        loss = v_loss + p_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    net = TinyNetwork()
    trainer = AlphaZeroTrainer(net, simulations=5)
    loss = trainer.train_step()
    print("training loss", loss)
