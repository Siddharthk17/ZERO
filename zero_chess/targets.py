"""Pure-Python asymmetric training target helpers."""

from __future__ import annotations

from .constants import WHITE

WIN_VALUE = 1.0
DRAW_VALUE = -1.0
LOSS_VALUE = -3.0
CONTEMPT_BONUS = 0.3
DRAW_BAND = 0.1
AGGRESSION_WEIGHT = 0.05
MOMENTUM_REWARD = 0.1
PANIC_PENALTY = -0.2


def opponent_value(value: float) -> float:
    """Convert a value from one side's perspective to the opponent's.

    The reward scale is asymmetric:
    win=+1, draw=-1, loss=-3. The affine transform below preserves those
    endpoints: +1 -> -3, -3 -> +1, and -1 -> -1.
    """
    return -float(value) - 2.0


def apply_contempt(value: float) -> float:
    """Apply the contempt factor to prevent settling for draws in even positions."""
    if -DRAW_BAND <= value <= DRAW_BAND:
        return value + CONTEMPT_BONUS
    return value


def game_result_to_value(result: str, perspective: int) -> float:
    """Map standard FEN outcome string to value scalar for a specific perspective."""
    if result == "1/2-1/2":
        return DRAW_VALUE
    return WIN_VALUE if (result == "1-0") == (perspective == WHITE) else LOSS_VALUE


def game_result_to_values(result: str) -> tuple[float, float]:
    """Convert a game result string into an asymmetric (white_reward, black_reward) tuple."""
    if result == "1/2-1/2":
        return (DRAW_VALUE, DRAW_VALUE)
    if result == "1-0":
        return (WIN_VALUE, LOSS_VALUE)
    return (LOSS_VALUE, WIN_VALUE)


def td_lambda_values(values: list[float], terminal: float, lam: float = 0.8) -> list[float]:
    """Calculate bootstrapped temporal difference targets across search states."""
    targets = []
    for idx, _ in enumerate(values):
        bootstrap = values[min(len(values) - 1, idx + 1)] if values else DRAW_VALUE
        targets.append(lam * terminal + (1.0 - lam) * opponent_value(bootstrap))
    return targets