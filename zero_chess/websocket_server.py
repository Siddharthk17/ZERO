"""WebSocket bridge for the ZERO UCI engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ZERO Engine WebSocket")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass(slots=True)
class UCIResult:
    move: str
    evaluation: float = 0.0
    nodes: int = 0


class UCIProcess:
    def __init__(self, command: list[str] | None = None) -> None:
        self.command = command or [
            sys.executable,
            "-m",
            "zero_chess.uci",
            "--checkpoint",
            "checkpoints/latest.pt",
            "--device",
            "cuda",
        ]
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()

    async def ensure(self) -> asyncio.subprocess.Process:
        if self.process and self.process.returncode is None:
            return self.process
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await self._send("uci")
        await self._read_until("uciok")
        await self._send("isready")
        await self._read_until("readyok")
        return self.process

    async def best_move(self, fen: str, move_time: int) -> UCIResult:
        async with self.lock:
            await self.ensure()
            await self._send(f"position fen {fen}")
            await self._send(f"go movetime {max(100, int(move_time))}")
            evaluation = 0.0
            nodes = 0
            while True:
                line = await self._read_line()
                if line.startswith("info "):
                    parsed_eval, parsed_nodes = parse_info(line)
                    evaluation = parsed_eval if parsed_eval is not None else evaluation
                    nodes = parsed_nodes if parsed_nodes is not None else nodes
                if line.startswith("bestmove "):
                    return UCIResult(line.split()[1], evaluation, nodes)

    async def _send(self, command: str) -> None:
        process = self.process
        if process is None or process.stdin is None:
            raise RuntimeError("UCI process is not running")
        process.stdin.write((command + "\n").encode())
        await process.stdin.drain()

    async def _read_line(self) -> str:
        process = self.process
        if process is None or process.stdout is None:
            raise RuntimeError("UCI process is not running")
        data = await process.stdout.readline()
        if not data:
            raise RuntimeError("UCI process exited")
        return data.decode(errors="replace").strip()

    async def _read_until(self, token: str) -> None:
        while True:
            if await self._read_line() == token:
                return


def parse_info(line: str) -> tuple[float | None, int | None]:
    parts = line.split()
    evaluation = None
    nodes = None
    if "nodes" in parts:
        try:
            nodes = int(parts[parts.index("nodes") + 1])
        except (ValueError, IndexError):
            nodes = None
    if "score" in parts:
        try:
            score_kind = parts[parts.index("score") + 1]
            score_value = int(parts[parts.index("score") + 2])
            evaluation = score_value / 100.0 if score_kind == "cp" else (100.0 if score_value > 0 else -100.0)
        except (ValueError, IndexError):
            evaluation = None
    return evaluation, nodes


engine = UCIProcess()


@app.get("/history")
def training_history(limit: int = 50) -> dict[str, list[dict]]:
    path = Path("data/training_games.jsonl")
    if not path.exists():
        return {"games": []}
    lines = path.read_text(encoding="utf-8").splitlines()
    selected = lines[-max(1, min(limit, 500)) :]
    games = []
    for line in selected:
        try:
            games.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    games.reverse()
    return {"games": games}


@app.websocket("/")
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_text()
            message = json.loads(payload)
            result = await engine.best_move(message["fen"], int(message.get("move_time", 1000)))
            await websocket.send_json({"move": result.move, "evaluation": result.evaluation, "nodes": result.nodes})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"move": "0000", "evaluation": 0.0, "nodes": 0, "error": str(exc)})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run ZERO WebSocket engine bridge.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--simulations", type=int, default=200)
    args = parser.parse_args(argv)
    engine.command = [
        sys.executable,
        "-m",
        "zero_chess.uci",
        "--checkpoint",
        args.checkpoint,
        "--device",
        args.device,
        "--simulations",
        str(args.simulations),
    ]
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
