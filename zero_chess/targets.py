"""Pure zero-sum training targets and Win-Draw-Loss (WDL) mappings.

All learned quantities in ZERO-X live in the closed interval ``[-1, 1]``.
Search-only contempt is deliberately kept out of these targets so it cannot
change the game-theoretic objective used by self-play or optimisation.
"""

from __future__ import annotations

WIN_VALUE = 1.0
DRAW_VALUE = 0.0
LOSS_VALUE = -1.0


def opponent_value(value: float) -> float:
    """Return the exact zero-sum value from the opponent's perspective."""
    return -float(value)


def apply_contempt(value: float, contempt: float = 0.10) -> float:
    """Apply a search-only draw bias without modifying network targets.

    This is for move selection and reporting only.  The result is clipped to
    the legal value domain, so callers never feed an invalid value back into a
    minimax backup.
    """
    value = float(value)
    if -0.15 <= value <= 0.15:
        return max(LOSS_VALUE, min(WIN_VALUE, value + float(contempt)))
    return value


def game_result_to_values(result: str) -> tuple[float, float]:
    """Map a PGN result to symmetric ``(white_value, black_value)`` targets."""
    if result == "1-0":
        return (WIN_VALUE, LOSS_VALUE)
    if result == "0-1":
        return (LOSS_VALUE, WIN_VALUE)
    if result == "1/2-1/2":
        return (DRAW_VALUE, DRAW_VALUE)
    raise ValueError(f"unsupported game result: {result!r}")


def terminal_wdl_target(value: float) -> tuple[float, float, float]:
    """Return a one-hot ``[win, draw, loss]`` terminal target."""
    if value > 0.5:
        return (1.0, 0.0, 0.0)
    if value < -0.5:
        return (0.0, 0.0, 1.0)
    return (0.0, 1.0, 0.0)
