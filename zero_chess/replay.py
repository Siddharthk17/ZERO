"""High-performance thread-safe prioritized replay buffer with hot and cold storage."""

from __future__ import annotations

import pickle
import random
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

def _get_available_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    return 128 * 1024 * 1024 * 1024

@dataclass(slots=True)
class Experience:
    """Represents a single evaluated chess board state for reinforcement training."""

    fen: str
    policy: dict[str, float]
    value: float
    td_value: float
    wdl: tuple[float, float, float]
    priority: float = 0.0
    value_prediction: float = 0.0
    importance_weight: float = 1.0
    reward_bonus: float = 0.0
    aggression_score: float = 0.0
    momentum_reward: float = 0.0
    panic_penalty: float = 0.0

    def __post_init__(self) -> None:
        self.fen = str(self.fen)
        policy_dict = {}
        for k, v in self.policy.items():
            k_str = str(k.uci() if hasattr(k, "uci") else k)
            policy_dict[k_str] = float(v.item() if hasattr(v, "item") else v)
        self.policy = policy_dict
        self.value = float(self.value.item() if hasattr(self.value, "item") else self.value)
        self.td_value = float(self.td_value.item() if hasattr(self.td_value, "item") else self.td_value)
        self.wdl = tuple(float(x.item() if hasattr(x, "item") else x) for x in self.wdl)
        self.priority = float(self.priority.item() if hasattr(self.priority, "item") else self.priority)
        self.value_prediction = float(self.value_prediction.item() if hasattr(self.value_prediction, "item") else self.value_prediction)
        self.importance_weight = float(self.importance_weight.item() if hasattr(self.importance_weight, "item") else self.importance_weight)
        self.reward_bonus = float(self.reward_bonus.item() if hasattr(self.reward_bonus, "item") else self.reward_bonus)
        self.aggression_score = float(self.aggression_score.item() if hasattr(self.aggression_score, "item") else self.aggression_score)
        self.momentum_reward = float(self.momentum_reward.item() if hasattr(self.momentum_reward, "item") else self.momentum_reward)
        self.panic_penalty = float(self.panic_penalty.item() if hasattr(self.panic_penalty, "item") else self.panic_penalty)


@dataclass(slots=True)
class SampleBatch:
    """A batch of sampled experiences with their indices and importance weights."""
    experiences: list[Experience]
    indices: list[int | None]
    weights: list[float]

class SumTree:
    """Binary prefix sum tree for O(log N) priority updates and sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity)

    @property
    def total(self) -> float:
        return self.tree[1]

    def update(self, index: int, priority: float) -> None:
        pos = index + self.capacity
        change = priority - self.tree[pos]
        self.tree[pos] = priority
        pos //= 2
        while pos:
            self.tree[pos] += change
            pos //= 2

    def get(self, value: float) -> int:
        idx = 1
        while idx < self.capacity:
            left = idx * 2
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - self.capacity

class PrioritizedReplayBuffer:
    """Thread-safe prioritized experience replay with automatic hardware memory adaptation."""

    def __init__(
        self,
        hot_capacity: int = 20_000,
        cold_path: str | Path | None = None,
        cold_capacity: int = 2_000_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 0.01,
        rng: random.Random | None = None,
    ) -> None:
        ram_bytes = _get_available_ram_bytes()
        # Protect tight systems (like 8GB RAM laptops) from crashing
        limit = 10_000 if ram_bytes < 2 * 1024 * 1024 * 1024 else 20_000
        self.hot_capacity = min(hot_capacity, limit)
        self.cold_path = Path(cold_path) if cold_path else None
        self.cold_capacity = cold_capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        
        self.hot: list[Experience] = []
        self._cursor = 0
        self._tree = SumTree(self.hot_capacity)
        self._max_priority = 1.0
        self._lock = threading.Lock()
        
        # FEN-based index mapping for collision-resistant priority updates
        self._fen_to_index: dict[str, int] = {}
        self._sqlite_conn: sqlite3.Connection | None = None
        self._cold_count: int = 0
        
        if self.cold_path is not None:
            self._init_cold()

    def __len__(self) -> int:
        with self._lock:
            return len(self.hot) + self._cold_count_unlocked()

    @property
    def hot_size(self) -> int:
        with self._lock:
            return len(self.hot)

    def add(self, exp: Experience) -> None:
        with self._lock:
            if exp.priority <= 0:
                exp.priority = self._max_priority
            else:
                exp.priority = self._priority_from_error(exp.priority)
                self._max_priority = max(self._max_priority, exp.priority)
                
            if len(self.hot) < self.hot_capacity:
                index = len(self.hot)
                self.hot.append(exp)
            else:
                index = self._cursor
                # Evict oldest hot experience to SQL cold storage
                old_exp = self.hot[index]
                self._fen_to_index.pop(old_exp.fen, None)
                self._append_cold_unlocked(old_exp)
                self.hot[index] = exp
                self._cursor = (self._cursor + 1) % self.hot_capacity
                
            self._fen_to_index[exp.fen] = index
            self._tree.update(index, exp.priority)

    def extend(self, experiences: list[Experience]) -> None:
        for exp in experiences:
            self.add(exp)

    def sample(self, batch_size: int) -> list[Experience]:
        return self.sample_with_weights(batch_size).experiences

    def sample_with_weights(self, batch_size: int, beta: float | None = None) -> SampleBatch:
        with self._lock:
            cold_count = self._cold_count_unlocked()
            if not self.hot and cold_count == 0:
                raise ValueError("cannot sample from an empty replay buffer")
                
            beta = self.beta if beta is None else beta
            hot_n = min(len(self.hot), int(round(batch_size * 0.8))) if self.hot else 0
            cold_n = batch_size - hot_n
            if cold_count == 0:
                hot_n = batch_size
                cold_n = 0
                
            experiences: list[Experience] = []
            indices: list[int | None] = []
            probs: list[float] = []
            total = max(self._tree.total, 1e-12)
            
            for _ in range(hot_n):
                value = self.rng.random() * total
                index = min(self._tree.get(value), len(self.hot) - 1)
                exp = self.hot[index]
                experiences.append(exp)
                indices.append(index)
                probs.append(max(exp.priority / total, 1e-12))
                
            for exp in self._sample_cold_unlocked(cold_n):
                experiences.append(exp)
                indices.append(None)
                probs.append(1.0 / max(1, cold_count))
                
            weights = [(max(1, len(self.hot)) * prob) ** (-beta) for prob in probs]
            max_weight = max(weights) if weights else 1.0
            weights = [weight / max_weight for weight in weights]
            
            for exp, weight in zip(experiences, weights, strict=True):
                exp.importance_weight = weight
                
            return SampleBatch(experiences, indices, weights)

    def update_priorities(self, indices_or_experiences, td_errors) -> None:
        with self._lock:
            for item, error in zip(indices_or_experiences, td_errors, strict=True):
                priority = self._priority_from_error(float(error))
                self._max_priority = max(self._max_priority, priority)
                
                if isinstance(item, int):
                    if 0 <= item < len(self.hot):
                        self.hot[item].priority = priority
                        self._tree.update(item, priority)
                elif isinstance(item, Experience):
                    item.priority = priority
                    index = self._fen_to_index.get(item.fen)
                    if index is not None and 0 <= index < len(self.hot):
                        self._tree.update(index, priority)

    def anneal_beta(self, step: int, total_steps: int = 500_000) -> float:
        self.beta = min(1.0, 0.4 + 0.6 * max(0, step) / max(1, total_steps))
        return self.beta

    def save(self, path: str | Path) -> None:
        with self._lock:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("wb") as fh:
                pickle.dump(
                    {
                        "hot": self.hot,
                        "cursor": self._cursor,
                        "max_priority": self._max_priority,
                    },
                    fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            tmp_path.replace(path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "PrioritizedReplayBuffer":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        replay = cls(**kwargs)
        replay.hot = payload["hot"]
        replay._cursor = payload["cursor"]
        replay._max_priority = payload.get("max_priority", 1.0)
        
        if len(replay.hot) > replay.hot_capacity:
            overflow = replay.hot[replay.hot_capacity:]
            replay.hot = replay.hot[:replay.hot_capacity]
            for exp in overflow:
                replay._append_cold_unlocked(exp)
            replay._cursor = replay._cursor % replay.hot_capacity
            
        replay._tree = SumTree(replay.hot_capacity)
        replay._fen_to_index.clear()
        for idx, exp in enumerate(replay.hot):
            replay._fen_to_index[exp.fen] = idx
            replay._tree.update(idx, exp.priority or replay._max_priority)
        return replay

    def _priority_from_error(self, error: float) -> float:
        return (abs(error) + self.epsilon) ** self.alpha

    def _init_cold(self) -> None:
        assert self.cold_path is not None
        self.cold_path.parent.mkdir(parents=True, exist_ok=True)
        # Persistent thread-safe connection with check_same_thread disabled since self._lock handles synchronization
        self._sqlite_conn = sqlite3.connect(self.cold_path, check_same_thread=False)
        # Write-Ahead Logging mode avoids write blocking
        self._sqlite_conn.execute("PRAGMA journal_mode = WAL")
        self._sqlite_conn.execute("PRAGMA synchronous = NORMAL")
        self._sqlite_conn.execute(
            "CREATE TABLE IF NOT EXISTS experiences (id INTEGER PRIMARY KEY AUTOINCREMENT, payload BLOB NOT NULL)"
        )
        self._sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_experiences_id ON experiences(id)")
        self._sqlite_conn.commit()
        self._cold_count = int(self._sqlite_conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0])

    def _append_cold_unlocked(self, exp: Experience) -> None:
        if self._sqlite_conn is None:
            return
        self._sqlite_conn.execute(
            "INSERT INTO experiences(payload) VALUES (?)",
            (pickle.dumps(exp, protocol=pickle.HIGHEST_PROTOCOL),)
        )
        self._sqlite_conn.commit()
        self._cold_count += 1

        # Enforce maximum sqlite buffer limits
        overflow = self._cold_count - self.cold_capacity
        if overflow > 0:
            self._sqlite_conn.execute(
                "DELETE FROM experiences WHERE id IN (SELECT id FROM experiences ORDER BY id LIMIT ?)",
                (overflow,),
            )
            self._sqlite_conn.commit()
            self._cold_count -= overflow

    def _sample_cold_unlocked(self, count: int) -> list[Experience]:
        if count <= 0 or self._sqlite_conn is None or self._cold_count <= 0:
            return []
        # For small tables, ORDER BY RANDOM() is fast enough
        if self._cold_count <= 10_000:
            rows = self._sqlite_conn.execute(
                "SELECT payload FROM experiences ORDER BY RANDOM() LIMIT ?", (count,)
            ).fetchall()
            return [pickle.loads(row[0]) for row in rows]
        # For large tables, random-ID probing is O(count*log n) vs O(n) for ORDER BY RANDOM()
        min_id, max_id = self._sqlite_conn.execute("SELECT MIN(id), MAX(id) FROM experiences").fetchone()
        if min_id is None or max_id is None or max_id <= 0:
            return []
        results: list[Experience] = []
        seen: set[int] = set()
        attempts = 0
        max_attempts = count * 4
        while len(results) < count and attempts < max_attempts:
            rid = self.rng.randint(min_id, max_id)
            if rid in seen:
                attempts += 1
                continue
            seen.add(rid)
            row = self._sqlite_conn.execute("SELECT payload FROM experiences WHERE id = ?", (rid,)).fetchone()
            if row is not None:
                results.append(pickle.loads(row[0]))
            attempts += 1
        return results

    def _cold_count_unlocked(self) -> int:
        return self._cold_count