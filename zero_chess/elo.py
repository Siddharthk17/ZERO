"""Running Elo-style rating helpers for ZERO."""

from __future__ import annotations

from .constants import BLACK, WHITE

DEFAULT_ELO = 0.0
DEFAULT_K = 32.0
ELO_FLOOR = -10_000.0


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def result_score(result: str, perspective: int) -> float:
    if result == "1/2-1/2":
        return 0.5
    if perspective == WHITE:
        return 1.0 if result == "1-0" else 0.0
    if perspective == BLACK:
        return 1.0 if result == "0-1" else 0.0
    raise ValueError(f"invalid perspective: {perspective}")


def update_rating(
    rating: float,
    opponent_rating: float,
    score: float,
    k: float = DEFAULT_K,
    floor: float = ELO_FLOOR,
) -> tuple[float, float]:
    expected = expected_score(rating, opponent_rating)
    new_rating = max(floor, rating + k * (score - expected))
    return new_rating, new_rating - rating


def update_rating_from_result(
    rating: float,
    opponent_rating: float,
    result: str,
    perspective: int,
    k: float = DEFAULT_K,
    floor: float = ELO_FLOOR,
) -> tuple[float, float]:
    if result == "1/2-1/2":
        win_gain = k * (1.0 - expected_score(rating, opponent_rating))
        new_rating = max(floor, rating - 0.5 * win_gain)
        return new_rating, new_rating - rating
    return update_rating(rating, opponent_rating, result_score(result, perspective), k, floor)
