import sys
import chess

sys.path.append("src")
from state import State

def test_take_action_creates_new_state_and_keeps_original_fen():
    state = State()
    original_fen = state.board.fen()

    new_state = state.take_action("e2e4")

    assert state.board.fen() == original_fen

    expected_board = chess.Board()
    expected_board.push(chess.Move.from_uci("e2e4"))
    assert new_state.board.fen() == expected_board.fen()
