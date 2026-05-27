"""Checkpoint management for continuous training."""

from __future__ import annotations

import json
import shutil
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
    def __init__(self, directory: str | Path = "checkpoints", keep_last: int = 20, permanent_every: int = 50) -> None:
        self.directory = Path(directory)
        self.keep_last = keep_last
        self.permanent_every = permanent_every
        self.index_path = self.directory / "index.json"

    def save(
        self,
        model: ZeroNet,
        iteration: int,
        elo: float = 0.0,
        optimizer_state=None,
        metrics: dict[str, float] | None = None,
    ) -> CheckpointMeta:
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"zero_iter_{iteration:07d}.pt"
        path = self.directory / name
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
        index = self._read_index()
        if not index:
            return None
        return CheckpointMeta(**max(index, key=lambda item: item["iteration"]))

    def _read_index(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write_index(self, index: list[dict]) -> None:
        tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        tmp_path.replace(self.index_path)

    def _prune(self, index: list[dict]) -> None:
        protected = {item["path"] for item in index[-self.keep_last :]}
        protected.update(item["path"] for item in index if item["iteration"] % self.permanent_every == 0)
        for item in index:
            path = Path(item["path"])
            if str(path) not in protected and path.exists():
                path.unlink()
