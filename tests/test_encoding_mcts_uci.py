from zero_chess import Board
from zero_chess.encoding import POLICY_SIZE, legal_policy_indices
from zero_chess.mcts import MCTS
from zero_chess.uci import UCIEngine


def test_policy_indices_are_unique_for_position() -> None:
    board = Board.starting_position()
    indices = list(legal_policy_indices(board).values())
    assert len(indices) == 20
    assert len(set(indices)) == 20
    assert all(0 <= idx < POLICY_SIZE for idx in indices)


def test_mcts_returns_legal_move() -> None:
    board = Board.starting_position()
    result = MCTS(simulations=8).search(board)
    assert result.move in board.legal_moves()


def test_uci_position_command() -> None:
    engine = UCIEngine()
    assert engine.handle("position startpos moves e2e4 e7e5")
    assert engine.board.fen().startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3")


def test_uci_core_commands(capsys) -> None:
    engine = UCIEngine()
    assert engine.handle("uci")
    assert engine.handle("isready")
    assert engine.handle("setoption name Simulations value 3")
    assert engine.handle("ucinewgame")
    assert engine.handle("position fen 8/8/8/8/8/8/8/K6k w - - 0 1")
    assert engine.handle("go depth 1")
    assert engine.handle("stop")
    assert engine.handle("quit") is False
    out = capsys.readouterr().out
    assert "uciok" in out
    assert "readyok" in out
    assert "bestmove" in out
