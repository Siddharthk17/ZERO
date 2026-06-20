"""High-performance, memory-safe Arena evaluation between checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from .board import Board
from .constants import BLACK, WHITE
from .elo import DEFAULT_ELO, update_rating_from_result
from .mcts import MCTS, NetworkEvaluator, UniformEvaluator

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
    """Play a series of games between two evaluators and return summary statistics.

    Each evaluator plays half the games as White.  Results are logged as JSON lines
    to ``log_path`` when provided.  Returns a dict with scores, win counts, draws,
    a ``promote`` flag (>60% score), and updated Elo ratings.
    """
    score_a = 0.0
    wins_a = wins_b = draws = 0
    current_elo_a = float(elo_a)
    current_elo_b = float(elo_b)

    for game_idx in range(games):
        board = Board.starting_position()
        a_is_white = game_idx < games // 2
        
        # Instantiate persistent MCTS objects for the game to leverage transposition table & subtree reuse
        mcts_a = MCTS(evaluator_a, simulations=simulations, resign_threshold=-1.0, batch_size=32)
        mcts_b = MCTS(evaluator_b, simulations=simulations, resign_threshold=-1.0, batch_size=32)

        for _ in range(max_plies):
            result = board.outcome()
            if result is not None:
                break
            
            use_a = (board.turn == WHITE and a_is_white) or (board.turn == BLACK and not a_is_white)
            active_mcts = mcts_a if use_a else mcts_b
            
            # Non-Zero-Sum aggressive search
            search = active_mcts.search(board, temperature=0.0)
            move = search.move
            if move is None:
                break
                
            board.push(move)
            
            # Advance tree roots for both players to reuse evaluated subtrees
            mcts_a.advance_to(move)
            mcts_b.advance_to(move)

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

        # Clear MCTS trees and garbage collect to prevent RAM build-up on tight systems (8GB)
        mcts_a.reset()
        mcts_b.reset()
        del mcts_a
        del mcts_b
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    result_summary = {
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
            fh.write(json.dumps(result_summary, sort_keys=True) + "\n")
            
    return result_summary

def _load_eval(path: str | None, device: str):
    if not path or path == "uniform":
        return UniformEvaluator()
    from .model import load_model

    return NetworkEvaluator(load_model(path, device), device)

def main(argv: list[str] | None = None) -> None:
    """CLI entry point: evaluate two checkpoints head-to-head."""
    parser = argparse.ArgumentParser(description="Evaluate two ZERO checkpoints.")
    parser.add_argument("--a", default="uniform")
    parser.add_argument("--b", default="uniform")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    
    result = play_arena(
        _load_eval(args.a, args.device), 
        _load_eval(args.b, args.device), 
        args.games, 
        args.simulations
    )
    result["win_rate_a"] = result["student_score"] / max(1, result["games"])
    print(result)

if __name__ == "__main__":  # pragma: no cover
    main()