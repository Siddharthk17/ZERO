import pytest

from zero_chess import Board
from zero_chess.encoding import terminal_wdl
from zero_chess.mcts import MCTS, UniformEvaluator
from zero_chess.targets import (
    DRAW_VALUE,
    LOSS_VALUE,
    WIN_VALUE,
    apply_contempt,
    game_result_to_values,
    opponent_value,
)
from zero_chess.uci import UCIEngine


def game_result_to_value(result: str, perspective: int) -> float:
    return game_result_to_values(result)[perspective]



def test_outcome_values_are_pure_zero_sum() -> None:
    assert game_result_to_value("1-0", 0) == WIN_VALUE
    assert game_result_to_value("1-0", 1) == LOSS_VALUE
    assert game_result_to_value("0-1", 0) == LOSS_VALUE
    assert game_result_to_value("1/2-1/2", 0) == DRAW_VALUE
    assert Board.from_fen("8/8/8/8/8/8/8/K6k w - - 0 1").result_value(0) == DRAW_VALUE
    assert terminal_wdl(DRAW_VALUE) == (0.0, 1.0, 0.0)


def test_zero_sum_opponent_value_transform() -> None:
    assert opponent_value(WIN_VALUE) == LOSS_VALUE
    assert opponent_value(LOSS_VALUE) == WIN_VALUE
    assert opponent_value(DRAW_VALUE) == DRAW_VALUE


def test_mcts_draw_target_and_search_only_contempt() -> None:
    draw = Board.from_fen("8/8/8/8/8/8/8/K6k w - - 0 1")
    result = MCTS(UniformEvaluator(), add_noise=False).search(draw, num_simulations=1, add_noise=False)
    assert result.root.q == DRAW_VALUE
    assert apply_contempt(0.0) == pytest.approx(0.1)

    start = Board()
    searched = MCTS(UniformEvaluator(), add_noise=False).search(start, num_simulations=1, add_noise=False)
    assert searched.root.visit_count > 0


def test_uci_time_pressure_aggression() -> None:
    engine = UCIEngine()
    more_time = engine._time_to_use(["wtime", "80000", "btime", "40000", "winc", "1000", "binc", "1000"])
    equal_standard = int(80000 / 40 + 1000 * 0.8)
    assert more_time == int(equal_standard * 1.2)

    less_time = engine._time_to_use(["wtime", "1000", "btime", "80000", "winc", "0", "binc", "0"])
    assert less_time == 150

