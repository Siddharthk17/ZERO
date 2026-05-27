"""Running Elo-style rating helpers for the ZERO training league."""

from __future__ import annotations

from .constants import BLACK, WHITE

DEFAULT_ELO = 0.0
DEFAULT_K = 32.0
ELO_FLOOR = -10_000.0


def expected_score(rating: float, opponent_rating: float) -> float:
    """Calculate the expected score of a player against a given opponent rating."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def result_score(result: str, perspective: int) -> float:
    """Map a result string ('1-0', '0-1', '1/2-1/2') to a score from a player's perspective."""
    if result == "1/2-1/2":
        return 0.5
    if perspective == WHITE:
        return 1.0 if result == "1-0" else 0.0
    if perspective == BLACK:
        return 1.0 if result == "0-1" else 0.0
    raise ValueError(f"invalid result or perspective: result={result!r}, perspective={perspective}")


def update_rating(
    rating: float,
    opponent_rating: float,
    score: float,
    k: float = DEFAULT_K,
    floor: float = ELO_FLOOR,
) -> tuple[float, float]:
    """Calculate and return the new rating and the rating delta."""
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
    """Update ratings based on game results, penalizing both players by half of win gain on draws."""
    if result == "1/2-1/2":
        win_gain = k * (1.0 - expected_score(rating, opponent_rating))
        new_rating = max(floor, rating - 0.5 * win_gain)
        return new_rating, new_rating - rating
    return update_rating(rating, opponent_rating, result_score(result, perspective), k, floor)