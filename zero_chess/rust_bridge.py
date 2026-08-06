"""Production bridge between Rust self-play generation and Python training."""

from __future__ import annotations

import importlib
import json
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .encoding import POLICY_SIZE
from .replay import Experience, PrioritizedReplayBuffer


class RustEngineUnavailableError(RuntimeError):
    """Raised when the native extension has not been built."""


_HISTORY_LOCK = threading.Lock()


def _engine():
    # Loading PyTorch first makes its bundled LibTorch shared libraries
    # available to the native extension on Linux wheels.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    try:
        module = importlib.import_module("zero_rust_engine")
    except ImportError as exc:
        try:
            module = importlib.import_module("zero_chess.zero_rust_engine")
        except ImportError:
            raise RustEngineUnavailableError(
                "zero_rust_engine is required for self-play; build with "
                "`cargo build --release --features libtorch,python-extension`"
            ) from exc
    return module


def generate_rust_self_play(
    model_path: str,
    *,
    num_games: int,
    simulations: int = 160,
    batch_size: int = 256,
    device: str = "cuda",
    seed: int | None = None,
) -> Mapping[str, Any]:
    if not 1 <= batch_size <= 256:
        raise ValueError("Rust evaluator batch_size must be in 1..=256")
    kwargs = {
        "model_path": model_path,
        "num_games": int(num_games),
        "simulations": int(simulations),
        "batch_size": int(batch_size),
        "device": device,
    }
    if seed is not None:
        kwargs["seed"] = int(seed)
    return _engine().generate_self_play_batch_rust(**kwargs)


def sparse_policy_by_index(
    indices: Iterable[int],
    values: Iterable[float],
) -> dict[int, float]:
    policy = {
        int(index): max(0.0, float(value))
        for index, value in zip(indices, values, strict=True)
        if 0 <= int(index) < POLICY_SIZE and float(value) > 0.0
    }
    total = sum(policy.values())
    return {index: probability / total for index, probability in policy.items()} if total else {}


def sparse_policy_to_uci(board, indices: Iterable[int], values: Iterable[float]) -> dict[str, float]:
    """Convert a sparse index policy to a ``{uci_move: probability}`` mapping.

    Only indices that correspond to legal moves in ``board`` are retained, so a
    corrupted or out-of-plane Rust policy can never inject an illegal move target.
    """
    from .encoding import move_to_policy_index

    legal_by_index: dict[int, str] = {}
    for move in board.legal_moves():
        legal_by_index[move_to_policy_index(board, move)] = move.uci()

    policy = sparse_policy_by_index(indices, values)
    uci_policy = {
        legal_by_index[index]: probability
        for index, probability in policy.items()
        if index in legal_by_index
    }
    total = sum(uci_policy.values())
    return {move: prob / total for move, prob in uci_policy.items()} if total else {}


def ingest_rust_batch(replay: PrioritizedReplayBuffer, payload: Mapping[str, Any]) -> int:
    inserted = 0
    for game in payload.get("games", []):
        for raw in game.get("experiences", []):
            policy = sparse_policy_by_index(
                raw.get("policy_indices", ()),
                raw.get("policy_values", ()),
            )
            wdl = tuple(float(value) for value in raw["wdl"])
            if len(wdl) != 3:
                raise ValueError("Rust self-play returned an invalid WDL target")
            q_mcts = float(raw.get("q_mcts", 0.0))
            raw_mat = raw.get("material", (0.0, 0.0))
            material = (
                (float(raw_mat[0]), float(raw_mat[1]))
                if isinstance(raw_mat, (tuple, list)) and len(raw_mat) == 2
                else (0.0, 0.0)
            )
            replay.add(
                Experience(
                    fen=raw["fen"],
                    policy=policy,
                    wdl=wdl,
                    target_kind=str(raw.get("target_kind", "terminal")),
                    q_mcts=q_mcts,
                    material=material,
                    moves_left=float(raw.get("moves_left", 0.0)),
                    opponent_policy=sparse_policy_by_index(
                        raw.get("opponent_policy_indices", ()),
                        raw.get("opponent_policy_values", ()),
                    ),
                    opponent_legal_policy=tuple(int(index) for index in raw.get("opponent_legal_indices", ())),
                    priority=1.0,
                    history_fens=tuple(raw.get("history_fens", ())),
                    repetitions=int(raw.get("repetitions", 1)),
                    history_repetitions=tuple(raw.get("history_repetitions", ())),
                )
            )
            inserted += 1
    return inserted


def append_rust_game_history(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    model_path: str,
    batch_index: int,
) -> int:
    """Persist native game provenance without placing it in training targets."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for game_index, game in enumerate(payload.get("games", [])):
        raw_result = float(game.get("result", 0.0))
        result = "1-0" if raw_result > 0.0 else "0-1" if raw_result < 0.0 else "1/2-1/2"
        experiences = list(game.get("experiences", []))
        target_counts: dict[str, int] = {}
        for experience in experiences:
            kind = str(experience.get("target_kind", "terminal"))
            target_counts[kind] = target_counts.get(kind, 0) + 1
        records.append(
            {
                "batch_index": int(batch_index),
                "game_index": int(game_index),
                "result": result,
                "termination": str(game.get("termination", "unknown")),
                "moves": list(game.get("moves", [])),
                "plies": len(game.get("moves", [])),
                "experiences": len(experiences),
                "target_counts": target_counts,
                "model_path": str(model_path),
            }
        )
    if not records:
        return 0
    with _HISTORY_LOCK:
        with path.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    return len(records)
