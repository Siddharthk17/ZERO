"""Thread-safe, self-healing checkpoint management for continuous training."""

from __future__ import annotations

import json
import re
import shutil
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .model import ZeroNet, save_model


@dataclass(slots=True)
class CheckpointMeta:
    iteration: int
    elo: float
    path: str
    created_at: str
    metrics: dict[str, float]


class CheckpointManager:
    """Manages atomic saves, thread-safe indexing, and automatic self-healing pruned file states."""

    def __init__(self, directory: str | Path = "checkpoints", keep_last: int = 20, permanent_every: int = 50) -> None:
        self.directory = Path(directory)
        self.keep_last = keep_last
        self.permanent_every = permanent_every
        self.index_path = self.directory / "index.json"
        self._lock = threading.RLock()

    def save(
        self,
        model: ZeroNet,
        iteration: int,
        elo: float = 0.0,
        optimizer_state=None,
        metrics: dict[str, float] | None = None,
    ) -> CheckpointMeta:
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            name = f"zero_iter_{iteration:07d}.pt"
            path = self.directory / name
            
            # Atomic weight write using the robust tmp-replace design
            save_model(path, model, iteration=iteration, elo=elo, optimizer=optimizer_state, metrics=metrics or {})
            
            latest_path = self.directory / "latest.pt"
            latest_tmp = latest_path.with_suffix(latest_path.suffix + ".tmp")
            shutil.copy2(path, latest_tmp)
            latest_tmp.replace(latest_path)
            
            meta = CheckpointMeta(
                iteration=iteration,
                elo=elo,
                path=str(path),
                created_at=datetime.now(timezone.utc).isoformat(),
                metrics=metrics or {},
            )
            
            index = self._read_index()
            index.append(asdict(meta))
            index.sort(key=lambda item: item["iteration"])
            
            self._write_index(index)
            self._prune(index)
            return meta

    def latest(self) -> CheckpointMeta | None:
        with self._lock:
            index = self._read_index()
            if not index:
                return None
            
            # Ensure the latest returned checkpoint file actually exists on disk
            for item in reversed(index):
                if Path(item["path"]).exists():
                    return CheckpointMeta(**item)
            return None

    def _read_index(self) -> list[dict]:
        with self._lock:
            if not self.index_path.exists():
                return self._reconstruct_index()
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                # Automatic Self-Healing: Rebuild database if corrupted
                return self._reconstruct_index()

    def _reconstruct_index(self) -> list[dict]:
        """Scans filesystem and repairs the index schema if index.json is corrupt or deleted."""
        with self._lock:
            index = []
            if not self.directory.exists():
                return index
                
            pattern = re.compile(r"zero_iter_(\d+)\.pt")
            for p in self.directory.glob("zero_iter_*.pt"):
                match = pattern.match(p.name)
                if match:
                    iter_num = int(match.group(1))
                    meta = CheckpointMeta(
                        iteration=iter_num,
                        elo=0.0,
                        path=str(p),
                        created_at=datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
                        metrics={},
                    )
                    index.append(asdict(meta))
                    
            index.sort(key=lambda item: item["iteration"])
            if index:
                try:
                    self._write_index(index)
                except OSError:
                    pass
            return index

    def _write_index(self, index: list[dict]) -> None:
        with self._lock:
            tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
            tmp_path.replace(self.index_path)

    def _prune(self, index: list[dict]) -> None:
        with self._lock:
            # Mark newest checkpoints and permanent multiples as protected
            protected = {item["path"] for item in index[-self.keep_last :]}
            protected.update(item["path"] for item in index if item["iteration"] % self.permanent_every == 0)
            
            for item in index:
                path = Path(item["path"])
                if str(path) not in protected and path.exists():
                    try:
                        path.unlink()
                    except OSError:
                        pass