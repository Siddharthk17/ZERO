"""Pure-Python asymmetric training target helpers with aggressive checkmate-focused rewards."""

from __future__ import annotations

# Custom Aggressive RL Anchor Values
CHECKMATE_WIN = 1.0
CHECKMATE_LOSS = -3.0
RESIGNATION_WIN = 0.0
RESIGNATION_LOSS = -30.0
STALEMATE_DRAW = -10.0
MAX_PLIES_DRAW = -20.0
STANDARD_DRAW = -1.0

# Critical Reinforcement Learning & Search Anchors [1]
DRAW_VALUE = -1.0
DRAW_BAND = 0.1
CONTEMPT_BONUS = 0.3
AGGRESSION_WEIGHT = 0.05
MOMENTUM_REWARD = 0.1
PANIC_PENALTY = -0.2

# Sorted anchor points (x, f(x)) where x is active player's payoff, f(x) is opponent's payoff
ANCHORS = [
    (-30.0, 0.0),       # I resign
    (-20.0, -20.0),     # Max plies draw
    (-10.0, -10.0),     # Stalemate
    (-3.0, 1.0),        # I lose by checkmate
    (-1.0, -1.0),       # Standard draw
    (0.0, -30.0),       # Opponent resigns
    (1.0, -3.0),        # I win by checkmate
]

def opponent_value(value: float) -> float:
    """Convert a value from one side's perspective to the opponent's using symmetric piecewise linear interpolation."""
    val = float(value)
    
    # Clip input value to the supported domain
    if val <= -30.0:
        return 0.0
    if val >= 1.0:
        return -3.0
        
    # Piecewise linear interpolation
    for i in range(len(ANCHORS) - 1):
        x0, y0 = ANCHORS[i]
        x1, y1 = ANCHORS[i + 1]
        if x0 <= val <= x1:
            return y0 if (x1 - x0) == 0 else y0 + (val - x0) * (y1 - y0) / (x1 - x0)
            
    return -29.0 - val  # Safe continuous fallback

def apply_contempt(value: float) -> float:
    """Apply the contempt factor to prevent settling for draws in even positions."""
    if -0.1 <= value <= 0.1:
        return value + 0.3
    return value

def game_result_to_values(result: str) -> tuple[float, float]:
    """Convert a game result string into an asymmetric (white_reward, black_reward) tuple."""
    if result == "1/2-1/2":
        return (-1.0, -1.0)
    if result == "1-0":
        return (1.0, -3.0)
    return (-3.0, 1.0)