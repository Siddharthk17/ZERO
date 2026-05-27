"""Arena evaluation between checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .board import Board
from .constants import BLACK, WHITE
from .elo import DEFAULT_ELO, update_rating, update_rating_from_result
from .mcts import MCTS, NetworkEvaluator, UniformEvaluator


def update_elo(elo_a: float, elo_b: float, score_a: float, games: int, k: float = 32.0) -> tuple[float, float]:
    score = score_a / max(1, games)
    new_elo_a, _ = update_rating(elo_a, elo_b, score, k * max(1, games))
    new_elo_b, _ = update_rating(elo_b, elo_a, 1.0 - score, k * max(1, games))
    return new_elo_a, new_elo_b


def play_arena(
    evaluator_a,
    evaluator_b,
    games: int = 40,
    simulations: int = 800,
    max_plies: int = 512,
    iteration: int = 0,
    elo_a: float = DEFAULT_ELO,
    elo_b: float = DEFAULT_ELO,
    log_path: str | None = "logs/arena.log",
) -> dict[str, float]:
    score_a = 0.0
    wins_a = wins_b = draws = 0
    current_elo_a = float(elo_a)
    current_elo_b = float(elo_b)
    for game_idx in range(games):
        board = Board.starting_position()
        a_is_white = game_idx < games // 2
        for _ in range(max_plies):
            result = board.outcome()
            if result is not None:
                break
            use_a = (board.turn == 0 and a_is_white) or (board.turn == 1 and not a_is_white)
            evaluator = evaluator_a if use_a else evaluator_b
            move = MCTS(evaluator, simulations=simulations, add_noise=False, resign_threshold=-1.0, batch_size=32).search(
                board, temperature=0.0, add_noise=False
            ).move
            if move is None:
                break
            board.push(move)
        result = board.outcome() or "1/2-1/2"
        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == a_is_white:
            score_a += 1.0
            wins_a += 1
        else:
            wins_b += 1
        a_perspective = WHITE if a_is_white else BLACK
        b_perspective = BLACK if a_is_white else WHITE
        previous_elo_a = current_elo_a
        previous_elo_b = current_elo_b
        current_elo_a, _ = update_rating_from_result(previous_elo_a, previous_elo_b, result, a_perspective)
        current_elo_b, _ = update_rating_from_result(previous_elo_b, previous_elo_a, result, b_perspective)
    result = {
        "iteration": iteration,
        "games": games,
        "student_score": score_a,
        "teacher_score": games - score_a,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "promote": score_a > games * 0.60,
        "elo_a": current_elo_a,
        "elo_b": current_elo_b,
    }
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, sort_keys=True) + "\n")
    return result


def _load_eval(path: str | None, device: str):
    if not path or path == "uniform":
        return UniformEvaluator()
    from .model import load_model

    return NetworkEvaluator(load_model(path, device), device)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate two ZERO checkpoints.")
    parser.add_argument("--a", default="uniform")
    parser.add_argument("--b", default="uniform")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = play_arena(_load_eval(args.a, args.device), _load_eval(args.b, args.device), args.games, args.simulations)
    result["win_rate_a"] = result["student_score"] / max(1, result["games"])
    print(result)


if __name__ == "__main__":  # pragma: no cover
    main()
