"""In-RAM prioritized replay for production self-play and training."""

from __future__ import annotations

import os
import pickle
import random
import threading
from copy import copy
from dataclasses import dataclass
from math import isfinite, nextafter
from pathlib import Path

import numpy as np

# Policy size matches the AlphaZero 73-plane encoding (73 * 64 = 4672).
_POLICY_SIZE = 73 * 64
_TARGET_KINDS = {"terminal", "truncated", "adjudicated"}


def _normalize_policy_keys(fen: str, policy: dict) -> dict[int, float]:
    """Accept integer policy indices or UCI move strings and return ``{int: float}``.

    The production Rust self-play path emits sparse integer indices into the 4672-dim
    AlphaZero policy space.  Legacy Python tests and the pure-Python MCTS path emit
    UCI move strings (e.g. ``"e2e4"``).  Both representations are accepted here: integer
    keys pass through directly (the fast path), while string keys are converted to
    policy indices by decoding the move against the position described by ``fen``.
    """
    if not policy:
        return {}
    from .board import Board
    from .encoding import move_to_policy_index
    from .move import Move

    board = Board.from_fen(fen)
    legal_by_uci = {move.uci(): move_to_policy_index(board, move) for move in board.legal_moves()}
    legal_indices = set(legal_by_uci.values())
    normalized: dict[int, float] = {}
    for key, value in policy.items():
        probability = float(value.item() if hasattr(value, "item") else value)
        if not isfinite(probability):
            raise ValueError("policy target contains a non-finite probability")
        if isinstance(key, str):
            move = Move.from_uci(key)
            index = legal_by_uci.get(move.uci())
        else:
            index = int(key)
        if index in legal_indices:
            normalized[index] = max(0.0, probability)
    return normalized


@dataclass(slots=True)
class Experience:
    """One tabula-rasa policy/WDL training sample."""

    fen: str
    policy: dict[int, float]
    wdl: tuple[float, float, float]
    target_kind: str = "terminal"
    q_mcts: float = 0.0
    material: tuple[float, float] = (0.0, 0.0)
    moves_left: float = 0.0
    opponent_policy: dict[int, float] | None = None
    opponent_legal_policy: tuple[int, ...] = ()
    priority: float = 0.0
    importance_weight: float = 1.0
    history_fens: tuple[str, ...] = ()
    repetitions: int = 1
    history_repetitions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.fen = str(self.fen)
        self.target_kind = str(self.target_kind)
        if self.target_kind not in _TARGET_KINDS:
            raise ValueError(f"unsupported replay target kind: {self.target_kind!r}")
        normalized_policy = _normalize_policy_keys(self.fen, self.policy)
        normalized_policy = {
            index: max(0.0, float(value))
            for index, value in normalized_policy.items()
            if isfinite(float(value))
        }
        self.policy = normalized_policy

        self.wdl = tuple(float(value.item() if hasattr(value, "item") else value) for value in self.wdl)
        if len(self.wdl) != 3:
            raise ValueError("WDL target must contain exactly win/draw/loss probabilities")
        if not all(isfinite(value) and value >= 0.0 for value in self.wdl):
            raise ValueError("WDL target must contain finite, non-negative probabilities")
        wdl_total = sum(self.wdl)
        if wdl_total <= 0.0:
            raise ValueError("WDL target must have positive probability mass")
        self.wdl = tuple(value / wdl_total for value in self.wdl)

        self.q_mcts = float(self.q_mcts.item() if hasattr(self.q_mcts, "item") else self.q_mcts)
        if not isfinite(self.q_mcts):
            raise ValueError("MCTS value target must be finite")
        self.q_mcts = min(1.0, max(-1.0, self.q_mcts))
        if isinstance(self.material, (list, tuple)) and len(self.material) == 2:
            white_material, black_material = self.material
            self.material = (
                min(1.0, max(0.0, float(white_material.item() if hasattr(white_material, "item") else white_material))),
                min(1.0, max(0.0, float(black_material.item() if hasattr(black_material, "item") else black_material))),
            )
        else:
            self.material = (0.0, 0.0)
        self.moves_left = min(1.0, max(0.0, float(
            self.moves_left.item() if hasattr(self.moves_left, "item") else self.moves_left
        )))
        if self.opponent_policy is not None:
            normalized_opponent_policy = {
                int(index): max(0.0, float(value.item() if hasattr(value, "item") else value))
                for index, value in self.opponent_policy.items()
                if 0 <= int(index) < 4_672
                and isfinite(float(value.item() if hasattr(value, "item") else value))
            }
            total = sum(normalized_opponent_policy.values())
            self.opponent_policy = (
                {index: value / total for index, value in normalized_opponent_policy.items()}
                if total > 0.0
                else None
            )
        self.opponent_legal_policy = tuple(
            sorted({int(index) for index in self.opponent_legal_policy if 0 <= int(index) < _POLICY_SIZE})
        )
        if self.opponent_policy is not None and self.opponent_legal_policy:
            legal_indices = set(self.opponent_legal_policy)
            filtered_opponent = {
                index: value for index, value in self.opponent_policy.items() if index in legal_indices
            }
            total = sum(filtered_opponent.values())
            self.opponent_policy = (
                {index: value / total for index, value in filtered_opponent.items()}
                if total > 0.0
                else None
            )
        self.priority = float(self.priority.item() if hasattr(self.priority, "item") else self.priority)
        if not isfinite(self.priority):
            raise ValueError("replay priority must be finite")
        self.importance_weight = float(
            self.importance_weight.item() if hasattr(self.importance_weight, "item") else self.importance_weight
        )
        if not isfinite(self.importance_weight):
            raise ValueError("importance weight must be finite")
        self.history_fens = tuple(str(fen) for fen in self.history_fens[:7])
        self.repetitions = max(1, min(255, int(self.repetitions)))
        history_repetitions = tuple(
            max(1, min(255, int(value))) for value in self.history_repetitions[:len(self.history_fens)]
        )
        self.history_repetitions = history_repetitions + (1,) * (len(self.history_fens) - len(history_repetitions))

    @property
    def value(self) -> float:
        return 0.5 * (self.wdl[0] - self.wdl[2]) + 0.5 * self.q_mcts


@dataclass(slots=True)
class SampleBatch:
    experiences: list[Experience]
    indices: list[int]
    weights: list[float]
    generations: list[int]


class SumTree:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._size = max(1, 1 << (capacity - 1).bit_length())
        self.tree = np.zeros(2 * self._size, dtype=np.float64)

    @property
    def total(self) -> float:
        return self.tree[1]

    def update(self, index: int, priority: float) -> None:
        if not 0 <= index < self.capacity:
            raise IndexError("sum-tree index out of range")
        pos = index + self._size
        change = priority - self.tree[pos]
        self.tree[pos] = priority
        pos //= 2
        while pos:
            self.tree[pos] += change
            pos //= 2

    def get(self, value: float) -> int:
        index = 1
        while index < self._size:
            left = index * 2
            if value < self.tree[left]:
                index = left
            else:
                value -= self.tree[left]
                index = left + 1
        return index - self._size


class PrioritizedReplayBuffer:
    def __init__(
        self,
        hot_capacity: int = 1_000_000,
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 0.01,
        rng: random.Random | None = None,
    ) -> None:
        if hot_capacity <= 0:
            raise ValueError("hot_capacity must be positive")
        self.hot_capacity = int(hot_capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.rng = rng or random.Random()
        self.hot: list[Experience] = []
        self._generations: list[int] = []
        self._cursor = 0
        self._tree = SumTree(self.hot_capacity)
        self._max_priority = 1.0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self.hot)

    def add(self, exp: Experience) -> None:
        with self._lock:
            priority = self._priority_from_error(exp.priority) if exp.priority > 0.0 else self._max_priority
            self._max_priority = max(self._max_priority, priority)
            if len(self.hot) < self.hot_capacity:
                index = len(self.hot)
                self.hot.append(exp)
                self._generations.append(0)
            else:
                index = self._cursor
                self.hot[index] = exp
                self._generations[index] += 1
                self._cursor = (self._cursor + 1) % self.hot_capacity
            exp.priority = priority
            self._tree.update(index, priority)

    def sample_with_weights(self, batch_size: int, beta: float | None = None) -> SampleBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with self._lock:
            if not self.hot:
                raise ValueError("cannot sample from an empty replay buffer")
            beta = self.beta if beta is None else float(beta)
            total = max(self._tree.total, 1e-12)
            experiences: list[Experience] = []
            indices: list[int] = []
            generations: list[int] = []
            probabilities: list[float] = []
            for _ in range(batch_size):
                value = min(self.rng.random() * total, nextafter(total, 0.0))
                index = self._tree.get(value)
                if not 0 <= index < len(self.hot):
                    raise RuntimeError("replay sum tree returned an inactive leaf")
                exp = self.hot[index]
                experiences.append(exp)
                indices.append(index)
                generations.append(self._generations[index])
                probabilities.append(max(exp.priority / total, 1e-12))
            weights = [(len(self.hot) * probability) ** (-beta) for probability in probabilities]
            maximum = max(weights, default=1.0)
            weights = [weight / maximum for weight in weights]
            for exp, weight in zip(experiences, weights, strict=True):
                exp.importance_weight = weight
            return SampleBatch(experiences, indices, weights, generations)

    def update_priorities(
        self,
        indices: list[int],
        errors: list[float],
        generations: list[int] | None = None,
    ) -> None:
        with self._lock:
            if generations is not None and len(generations) != len(indices):
                raise ValueError("generations must match indices length")
            for offset, (index, error) in enumerate(zip(indices, errors, strict=True)):
                if not 0 <= index < len(self.hot):
                    continue
                if generations is not None and generations[offset] != self._generations[index]:
                    continue
                priority = self._priority_from_error(float(error))
                self._max_priority = max(self._max_priority, priority)
                self.hot[index].priority = priority
                self._tree.update(index, priority)

    def anneal_beta(self, step: int, total_steps: int = 500_000) -> float:
        self.beta = min(1.0, 0.4 + 0.6 * max(0, step) / max(1, total_steps))
        return self.beta

    def _priority_from_error(self, error: float) -> float:
        return (abs(error) + self.epsilon) ** self.alpha

    # -- Compatibility API -------------------------------------------------

    @property
    def hot_size(self) -> int:
        """Number of experiences currently held in the hot (in-RAM) buffer."""
        with self._lock:
            return len(self.hot)

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample ``batch_size`` experiences by priority (without importance weights).

        Convenience wrapper for callers that only need the raw experiences.
        """
        return self.sample_with_weights(batch_size).experiences

    def extend(self, experiences: list[Experience]) -> None:
        """Add a batch of experiences sequentially."""
        for exp in experiences:
            self.add(exp)

    def save(self, path: str | Path) -> None:
        """Persist the hot buffer and priorities to ``path`` via pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                # Freeze each record before releasing the lock.  Training can
                # update priorities while the serialized snapshot is written.
                "hot": [copy(exp) for exp in self.hot],
                "hot_capacity": self.hot_capacity,
                "alpha": self.alpha,
                "beta": self.beta,
                "epsilon": self.epsilon,
                "_cursor": self._cursor,
                "_max_priority": self._max_priority,
                "_generations": self._generations.copy(),
                "rng_state": self.rng.getstate(),
            }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path, hot_capacity: int | None = None) -> "PrioritizedReplayBuffer":
        """Load a buffer previously written by :meth:`save`."""
        path = Path(path)
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        capacity = hot_capacity if hot_capacity is not None else data["hot_capacity"]
        buffer = cls(
            hot_capacity=capacity,
            alpha=data["alpha"],
            beta=data["beta"],
            epsilon=data["epsilon"],
        )
        saved_hot = list(data["hot"])
        for exp in saved_hot:
            if not hasattr(exp, "target_kind"):
                exp.target_kind = "terminal"
        saved_generations = list(data.get("_generations", [0] * len(saved_hot)))
        saved_capacity = int(data.get("hot_capacity", len(saved_hot)))
        if len(saved_generations) != len(saved_hot):
            saved_generations = [0] * len(saved_hot)
        if len(saved_hot) > capacity:
            if len(saved_hot) == saved_capacity:
                cursor = int(data.get("_cursor", 0)) % saved_capacity
                ordered_hot = saved_hot[cursor:] + saved_hot[:cursor]
                ordered_generations = saved_generations[cursor:] + saved_generations[:cursor]
            else:
                ordered_hot = saved_hot
                ordered_generations = saved_generations
            saved_hot = ordered_hot[-capacity:]
            saved_generations = ordered_generations[-capacity:]
        buffer.hot = saved_hot
        buffer._generations = saved_generations
        buffer._cursor = 0 if len(buffer.hot) >= capacity else len(buffer.hot)
        buffer._max_priority = data["_max_priority"]
        if data.get("rng_state") is not None:
            buffer.rng.setstate(data["rng_state"])
        # Rebuild the SumTree from stored priorities.
        for index, exp in enumerate(buffer.hot):
            buffer._tree.update(index, exp.priority)
        return buffer
