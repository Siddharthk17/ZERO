import pytest

from zero_chess import Board
from zero_chess.arena import play_arena
from zero_chess.encoding import terminal_wdl
from zero_chess.mcts import MCTS, UniformEvaluator
from zero_chess.self_play import (
    PositionRecord,
    _aggression_score,
    _finalize,
    _momentum_reward,
    _panic_penalty,
)
from zero_chess.targets import (
    AGGRESSION_WEIGHT,
    DRAW_VALUE,
    CHECKMATE_LOSS as LOSS_VALUE,
    MOMENTUM_REWARD,
    PANIC_PENALTY,
    CHECKMATE_WIN as WIN_VALUE,
    apply_contempt,
    game_result_to_values,
    opponent_value,
)
from zero_chess.uci import UCIEngine


def game_result_to_value(result: str, perspective: int) -> float:
    return game_result_to_values(result)[perspective]



def test_outcome_values_are_hunter_scaled() -> None:
    assert game_result_to_value("1-0", 0) == WIN_VALUE
    assert game_result_to_value("1-0", 1) == LOSS_VALUE
    assert game_result_to_value("0-1", 0) == LOSS_VALUE
    assert game_result_to_value("1/2-1/2", 0) == DRAW_VALUE
    assert Board.from_fen("8/8/8/8/8/8/8/K6k w - - 0 1").result_value(0) == DRAW_VALUE
    assert terminal_wdl(DRAW_VALUE) == (0.0, 1.0, 0.0)


def test_asymmetric_opponent_value_transform() -> None:
    assert opponent_value(WIN_VALUE) == LOSS_VALUE
    assert opponent_value(LOSS_VALUE) == WIN_VALUE
    assert opponent_value(DRAW_VALUE) == DRAW_VALUE


def test_mcts_draw_aversion_and_contempt() -> None:
    draw = Board.from_fen("8/8/8/8/8/8/8/K6k w - - 0 1")
    result = MCTS(UniformEvaluator(), add_noise=False).search(draw, num_simulations=1, add_noise=False)
    assert result.root.q == DRAW_VALUE
    assert apply_contempt(0.0) == pytest.approx(0.3)

    start = Board()
    searched = MCTS(UniformEvaluator(), add_noise=False).search(start, num_simulations=1, add_noise=False)
    assert searched.root.visit_count > 0


def test_self_play_reward_shaping_helpers() -> None:
    attacking = Board.from_fen("3rk3/8/8/8/3Q4/8/8/4K3 w - - 0 1")
    assert _aggression_score(attacking) > 0

    capture = Board.from_fen("4k3/8/8/8/3Q4/8/3p4/4K3 w - - 0 1")
    move = next(m for m in capture.legal_moves() if m.uci() == "d4d2")
    assert _momentum_reward(capture, move) == MOMENTUM_REWARD
    assert _panic_penalty(capture, move) == PANIC_PENALTY

    record = PositionRecord(capture.fen(), {move.uci(): 1.0}, capture.turn, 0.0, 0.5, MOMENTUM_REWARD, PANIC_PENALTY)
    exp = _finalize([record], "1-0", __import__("random").Random(1), augment=False)[0]
    assert exp.reward_bonus == pytest.approx(AGGRESSION_WEIGHT * 0.5 + MOMENTUM_REWARD + PANIC_PENALTY)


def test_uci_time_pressure_aggression() -> None:
    engine = UCIEngine()
    more_time = engine._time_to_use(["wtime", "80000", "btime", "40000", "winc", "1000", "binc", "1000"])
    equal_standard = int(80000 / 40 + 1000 * 0.8)
    assert more_time == int(equal_standard * 1.2)

    less_time = engine._time_to_use(["wtime", "1000", "btime", "80000", "winc", "0", "binc", "0"])
    assert less_time >= 500


def test_arena_draws_are_student_losses_and_promotion_needs_sixty_percent(monkeypatch) -> None:
    monkeypatch.setattr(Board, "outcome", lambda self: "1/2-1/2")
    result = play_arena(UniformEvaluator(), UniformEvaluator(), games=40, simulations=1, max_plies=0, log_path=None)
    assert result["student_score"] == 0.0
    assert result["elo_a"] < 0.0
    assert result["elo_b"] < 0.0
    assert not result["promote"]

    wins = iter(["1-0"] * 20 + ["0-1"] * 4 + ["1-0"] * 16)
    monkeypatch.setattr(Board, "outcome", lambda self: next(wins, "0-1"))
    result = play_arena(UniformEvaluator(), UniformEvaluator(), games=40, simulations=1, max_plies=0, log_path=None)
    assert result["student_score"] == 24.0
    assert not result["promote"]

    wins = iter(["1-0"] * 20 + ["0-1"] * 5 + ["1-0"] * 15)
    monkeypatch.setattr(Board, "outcome", lambda self: next(wins, "0-1"))
    result = play_arena(UniformEvaluator(), UniformEvaluator(), games=40, simulations=1, max_plies=0, log_path=None)
    assert result["student_score"] == 25.0
    assert result["promote"]


def test_self_play_main_cli(tmp_path) -> None:
    from zero_chess.self_play import main as self_play_main
    pgn_file = tmp_path / "test_selfplay.pgn"
    self_play_main([
        "--games", "1",
        "--simulations", "1",
        "--max-plies", "2",
        "--out-pgn", str(pgn_file)
    ])
    assert pgn_file.exists()
    assert pgn_file.read_text(encoding="utf-8") != ""

