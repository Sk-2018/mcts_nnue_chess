import argparse
import json
import os

from state import State
from mcts import mcts, nnue_policy


class SelfLearner:
    """Simple table-based learner storing centipawn evaluations by FEN."""

    def __init__(self, alpha=0.1, path="data/selfplay.json"):
        self.alpha = alpha
        self.path = path
        self.values = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    self.values = json.load(f)
                except json.JSONDecodeError:
                    self.values = {}

    def get_value(self, fen):
        return self.values.get(fen)

    def update(self, fens, result):
        # convert result (1/0.5/0) to centipawns
        target = (result - 0.5) * 20000
        for fen in fens:
            old = self.values.get(fen, 0.0)
            new = old + self.alpha * (target - old)
            self.values[fen] = new

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.values, f)


def play_game(learner, iterations=50):
    state = State()
    searcher = mcts(iterationLimit=iterations,
                    rolloutPolicy=lambda s: nnue_policy(s, learner))
    fens = []
    while not state.is_terminal():
        fens.append(state.board.fen())
        move, _ = searcher.search(state)
        state = state.take_action(move)
    fens.append(state.board.fen())

    res = state.board.result()
    if res == "1-0":
        result = 1.0
    elif res == "0-1":
        result = 0.0
    else:
        result = 0.5
    learner.update(fens, result)
    learner.save()
    return result


def main():
    parser = argparse.ArgumentParser(description="Self-play training loop")
    parser.add_argument("--games", type=int, default=10, help="number of games to play")
    parser.add_argument("--iterations", type=int, default=50,
                        help="MCTS iterations per move")
    args = parser.parse_args()

    learner = SelfLearner()
    for i in range(args.games):
        result = play_game(learner, iterations=args.iterations)
        print(f"Game {i+1}: result {result}")


if __name__ == "__main__":
    main()
