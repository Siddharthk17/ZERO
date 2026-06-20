"""Highly stable WebSocket bridge with self-healing subprocess recovery for the ZERO engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

@dataclass(slots=True)
class UCIResult:
    """Result from a UCI best-move query: move string, evaluation, and node count."""
    move: str
    evaluation: float = 0.0
    nodes: int = 0

class UCIProcess:
    """Manages an asynchronous, self-healing UCI subprocess wrapper."""

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
        """Verify the subprocess status, spinning up a fresh handle if dead."""
        if self.process and self.process.returncode is None:
            return self.process
        
        await self.close()
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
        """Acquire the best move from the engine, executing a transaction under lock."""
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

    async def close(self) -> None:
        """Atomic teardown of the background subprocess to release locked resources."""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

    async def _send(self, command: str) -> None:
        process = self.process
        if process is None or process.stdin is None:
            raise RuntimeError("UCI process is not running")
        try:
            process.stdin.write((command + "\n").encode())
            await process.stdin.drain()
        except Exception as exc:
            await self.close()
            raise RuntimeError("failed to transmit command to subprocess") from exc

    async def _read_line(self) -> str:
        process = self.process
        if process is None or process.stdout is None:
            raise RuntimeError("UCI process is not running")
        try:
            data = await process.stdout.readline()
            if not data:
                await self.close()
                raise RuntimeError("UCI process reached EOF")
            return data.decode(errors="replace").strip()
        except Exception as exc:
            await self.close()
            raise RuntimeError("failed to read stream from subprocess") from exc

    async def _read_until(self, token: str) -> None:
        while True:
            if await self._read_line() == token:
                return

def parse_info(line: str) -> tuple[float | None, int | None]:
    """Parse a UCI 'info' line and return (evaluation, nodes) or (None, None) if absent."""
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Safely terminate the background engine process when FastAPI shuts down."""
    yield
    await engine.close()

app = FastAPI(title="ZERO Engine WebSocket", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _tail_file_lines(path: Path, max_lines: int = 500) -> list[str]:
    """Optimized binary seek tail reader to prevent RAM exhaustion on large FEN logs."""
    lines: list[bytes] = []
    if not path.exists():
        return []
    
    chunk_size = 4096
    with path.open("rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        
        buffer = bytearray()
        pointer = file_size
        
        while pointer > 0 and len(lines) <= max_lines:
            pointer = max(0, pointer - chunk_size)
            f.seek(pointer)
            chunk = f.read(file_size - pointer if pointer == 0 else chunk_size)
            buffer[:0] = chunk
            
            while b"\n" in buffer:
                newline_index = buffer.rindex(b"\n")
                line = buffer[newline_index + 1:]
                if line:
                    lines.append(bytes(line))
                buffer = buffer[:newline_index]
                if len(lines) > max_lines:
                    break
            
            file_size = pointer
            
        if len(lines) <= max_lines and buffer:
            lines.append(bytes(buffer))
            
    lines.reverse()
    return [line.decode("utf-8") for line in lines[-max_lines:]]

@app.get("/history")
def training_history(limit: int = 50) -> dict[str, list[dict]]:
    """Return recent training game records from the JSONL log, without PGN payloads."""
    path = Path("data/training_games.jsonl")
    if not path.exists():
        return {"games": []}
    limit_val = max(1, min(limit, 500))
    selected = _tail_file_lines(path, limit_val)
    games = []
    for line in selected:
        try:
            record = json.loads(line)
            record.pop("pgn", None)
            games.append(record)
        except json.JSONDecodeError:
            continue
    games.reverse()
    return {"games": games}

@app.websocket("/")
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint: receive FEN positions, return best moves from the UCI engine."""
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_text()
            message = json.loads(payload)
            try:
                result = await engine.best_move(message["fen"], int(message.get("move_time", 1000)))
                await websocket.send_json({"move": result.move, "evaluation": result.evaluation, "nodes": result.nodes})
            except Exception as sub_exc:
                await engine.close()  # Force process recovery on transaction failures
                await websocket.send_json({"move": "0000", "evaluation": 0.0, "nodes": 0, "error": str(sub_exc)})
    except WebSocketDisconnect:
        return

def main(argv: list[str] | None = None) -> None:
    """CLI entry point: start the FastAPI WebSocket server bridging to the UCI engine."""
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