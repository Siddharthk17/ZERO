"""Deterministic checkpoint gating for the ZERO-X training loop."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .board import Board
from .constants import WHITE
from .mcts import MCTS, NetworkEvaluator
from .model import load_model


@dataclass(slots=True)
class MatchResult:
    games: int
    wins_a: int
    wins_b: int
    draws: int
    score_a: float
    score_fraction: float
    elo_difference: float
    score_low: float
    score_high: float
    simulations: int
    opening_random_plies: int
    seed: int

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def play_match(
    evaluator_a,
    evaluator_b,
    *,
    games: int = 40,
    simulations: int = 64,
    max_plies: int = 512,
    opening_random_plies: int = 4,
    seed: int = 0,
) -> MatchResult:
    """Play a color-balanced, no-noise match between two evaluators."""
    if games <= 0 or games % 2 != 0:
        raise ValueError("gating games must be a positive even number")
    if simulations <= 0 or max_plies <= 0 or opening_random_plies < 0:
        raise ValueError("gating simulations and max_plies must be positive")

    wins_a = wins_b = draws = 0
    rng = random.Random(seed)
    for game_index in range(games):
        board = Board.starting_position()
        history: list[Board] = []
        mcts_a = MCTS(evaluator_a, simulations=simulations, batch_size=32, add_noise=False, resign_threshold=-1.0)
        mcts_b = MCTS(evaluator_b, simulations=simulations, batch_size=32, add_noise=False, resign_threshold=-1.0)
        a_is_white = game_index < games // 2

        opening_count = min(opening_random_plies, max_plies)
        for _ in range(opening_count):
            if board.outcome() is not None:
                break
            legal = board.legal_moves()
            if not legal:
                break
            played = rng.choice(legal)
            previous = board.copy()
            board.push(played)
            history.insert(0, previous)
            del history[7:]

        for _ply in range(max_plies - opening_count):
            result = board.outcome()
            if result is not None:
                break
            active = mcts_a if (board.turn == WHITE) == a_is_white else mcts_b
            search = active.search(
                board,
                num_simulations=simulations,
                temperature=0.0,
                add_noise=False,
                history=history,
            )
            if search.move is None:
                break
            played = search.move
            previous = board.copy()
            board.push(played)
            history.insert(0, previous)
            del history[7:]
            mcts_a.advance_to(played)
            mcts_b.advance_to(played)

        result = board.outcome() or "1/2-1/2"
        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == a_is_white:
            wins_a += 1
        else:
            wins_b += 1

        # Keep the random stream advancing even when the game is deterministic
        # so future opening/match extensions remain seed-stable.
        rng.random()

    score_a = wins_a + 0.5 * draws
    score_fraction = score_a / games
    elo_difference = _elo_difference(score_fraction)
    score_low, score_high = _wilson_interval(score_fraction, games)
    return MatchResult(
        games=games,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
        score_a=score_a,
        score_fraction=score_fraction,
        elo_difference=elo_difference,
        score_low=score_low,
        score_high=score_high,
        simulations=simulations,
        opening_random_plies=opening_random_plies,
        seed=seed,
    )


def gate_checkpoints(
    candidate_path: str | Path,
    incumbent_path: str | Path,
    *,
    games: int = 40,
    simulations: int = 64,
    device: str = "cpu",
    opening_random_plies: int = 4,
    seed: int = 0,
    log_path: str | Path | None = None,
) -> MatchResult:
    """Evaluate candidate A against incumbent B without changing either file."""
    candidate = load_model(candidate_path, device)
    incumbent = load_model(incumbent_path, device)
    result = play_match(
        NetworkEvaluator(candidate, device),
        NetworkEvaluator(incumbent, device),
        games=games,
        simulations=simulations,
        opening_random_plies=opening_random_plies,
        seed=seed,
    )
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result.as_dict(), sort_keys=True) + "\n")
    return result


def _elo_difference(score_fraction: float) -> float:
    score_fraction = min(1.0 - 1e-6, max(1e-6, float(score_fraction)))
    return 400.0 * math.log10(score_fraction / (1.0 - score_fraction))


def _wilson_interval(score_fraction: float, games: int) -> tuple[float, float]:
    z = 1.959963984540054
    n = float(games)
    p = float(score_fraction)
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)
