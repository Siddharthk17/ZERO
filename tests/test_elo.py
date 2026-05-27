import pytest

from zero_chess.constants import BLACK, WHITE
from zero_chess.elo import DEFAULT_ELO, expected_score, result_score, update_rating_from_result


def test_default_elo_starts_at_zero() -> None:
    assert DEFAULT_ELO == 0.0


def test_result_score_uses_chess_result_from_perspective() -> None:
    assert result_score("1-0", WHITE) == 1.0
    assert result_score("0-1", WHITE) == 0.0
    assert result_score("0-1", BLACK) == 1.0
    assert result_score("1/2-1/2", WHITE) == 0.5


def test_elo_gain_loss_and_floor() -> None:
    rating, delta = update_rating_from_result(0.0, 0.0, "1-0", WHITE)
    assert rating == pytest.approx(16.0)
    assert delta == pytest.approx(16.0)

    rating, delta = update_rating_from_result(rating, rating, "0-1", WHITE)
    assert rating == pytest.approx(0.0)
    assert delta == pytest.approx(-16.0)

    rating, delta = update_rating_from_result(0.0, 0.0, "0-1", WHITE)
    assert rating == pytest.approx(-16.0)
    assert delta == pytest.approx(-16.0)

    rating, delta = update_rating_from_result(0.0, 0.0, "0-1", WHITE, floor=0.0)
    assert rating == 0.0
    assert delta == 0.0


def test_draw_penalizes_both_players_by_half_of_win_gain() -> None:
    white_rating, white_delta = update_rating_from_result(0.0, 0.0, "1/2-1/2", WHITE)
    black_rating, black_delta = update_rating_from_result(0.0, 0.0, "1/2-1/2", BLACK)
    assert white_rating == pytest.approx(-8.0)
    assert black_rating == pytest.approx(-8.0)
    assert white_delta == pytest.approx(-8.0)
    assert black_delta == pytest.approx(-8.0)


def test_expected_score_favors_higher_rating() -> None:
    assert expected_score(200.0, 0.0) > 0.5
    assert expected_score(0.0, 200.0) < 0.5
