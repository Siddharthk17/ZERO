"""Universal Chess Interface entry point."""

from __future__ import annotations

import argparse
import shlex
import sys
import threading
import time
from dataclasses import dataclass

from .board import Board
from .mcts import MCTS, NetworkEvaluator, SearchResult, UniformEvaluator


@dataclass(slots=True)
class UCIOptions:
    simulations: int = 200
    cpuct: float = 1.5
    checkpoint: str | None = None
    device: str = "cpu"


class UCIEngine:
    def __init__(self, options: UCIOptions | None = None) -> None:
        self.options = options or UCIOptions()
        self.board = Board()
        self.evaluator = UniformEvaluator()
        self.mcts = MCTS(self.evaluator, simulations=self.options.simulations, c_puct=self.options.cpuct, add_noise=False)
        self._search_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._best_result: SearchResult | None = None
        self._position_seen = False
        self._searched = False
        self._load_checkpoint()

    def loop(self) -> None:
        for line in sys.stdin:
            if not self.handle(line.strip()):
                return
        if self._position_seen and not self._searched:
            self._cmd_go(["nodes", "1"])

    def handle(self, line: str) -> bool:
        if not line:
            return True
        parts = shlex.split(line)
        cmd = parts[0]
        args = parts[1:]
        try:
            if cmd == "uci":
                self._cmd_uci()
            elif cmd == "isready":
                print("readyok", flush=True)
            elif cmd == "setoption":
                self._cmd_setoption(args)
            elif cmd == "ucinewgame":
                self.board = Board()
                self.mcts.reset()
            elif cmd == "position":
                self._cmd_position(args)
                self._position_seen = True
            elif cmd == "go":
                self._cmd_go(args)
            elif cmd == "stop":
                self._cmd_stop()
            elif cmd == "quit":
                self._cmd_stop(emit=False)
                return False
            elif cmd == "d":
                print(self.board.fen(), flush=True)
        except Exception as exc:
            print(f"info string error: {exc}", flush=True)
        return True

    def _cmd_uci(self) -> None:
        print("id name ZERO", flush=True)
        print("id author Sid", flush=True)
        print("option name Simulations type spin default 200 min 1 max 100000", flush=True)
        print("option name CPuct type string default 1.5", flush=True)
        print("option name Checkpoint type string default", flush=True)
        print("option name Device type combo default cpu var cpu var cuda", flush=True)
        print("uciok", flush=True)

    def _cmd_setoption(self, args: list[str]) -> None:
        text = " ".join(args)
        if " name " not in f" {text} ":
            return
        name_part, _, value_part = text.partition(" value ")
        name = name_part.replace("name ", "", 1).strip().lower()
        value = value_part.strip()
        if name == "simulations":
            self.options.simulations = int(value)
            self.mcts.simulations = self.options.simulations
        elif name == "cpuct":
            self.options.cpuct = float(value)
            self.mcts.c_puct = self.options.cpuct
        elif name == "checkpoint":
            self.options.checkpoint = value or None
            self._load_checkpoint()
        elif name == "device":
            self.options.device = value
            self._load_checkpoint()

    def _cmd_position(self, args: list[str]) -> None:
        if not args:
            return
        idx = 0
        if args[idx] == "startpos":
            self.board = Board()
            idx += 1
        elif args[idx] == "fen":
            fen_fields = []
            idx += 1
            while idx < len(args) and args[idx] != "moves":
                fen_fields.append(args[idx])
                idx += 1
            self.board = Board.from_fen(" ".join(fen_fields))
        if idx < len(args) and args[idx] == "moves":
            for move in args[idx + 1 :]:
                self.board.push_uci(move)
        self.mcts.reset()

    def _cmd_go(self, args: list[str]) -> None:
        self._searched = True
        if "infinite" in args:
            self._start_infinite_search()
            return
        if "movetime" in args:
            ms = int(args[args.index("movetime") + 1])
            result = self.mcts.search_time(self.board, ms, temperature=0.0, add_noise=False)
        else:
            simulations = self._simulations_for_go(args)
            result = self.mcts.search(self.board, num_simulations=simulations, temperature=0.0, add_noise=False)
        self._best_result = result
        self._emit_bestmove(result)

    def _start_infinite_search(self) -> None:
        self._stop_event.clear()
        self._best_result = None

        def worker() -> None:
            while not self._stop_event.is_set():
                self._best_result = self.mcts.search(self.board, num_simulations=self.mcts.batch_size, temperature=0.0, add_noise=False)

        self._search_thread = threading.Thread(target=worker, daemon=True)
        self._search_thread.start()

    def _cmd_stop(self, emit: bool = True) -> None:
        self._stop_event.set()
        if self._search_thread and self._search_thread.is_alive():
            self._search_thread.join(timeout=1.0)
        if emit:
            self._emit_bestmove(self._best_result)

    def _emit_bestmove(self, result: SearchResult | None) -> None:
        if result is None or result.move is None:
            legal = self.board.legal_moves()
            best = legal[0].uci() if legal else "0000"
            nodes = 0
        else:
            best = result.move.uci()
            nodes = sum(result.visits.values())
        print(f"info depth 1 nodes {nodes} score cp 0", flush=True)
        print(f"bestmove {best}", flush=True)

    def _simulations_for_go(self, args: list[str]) -> int:
        if "nodes" in args:
            return max(1, int(args[args.index("nodes") + 1]))
        if "depth" in args:
            return max(1, int(args[args.index("depth") + 1]) * 100)
        if "wtime" in args or "btime" in args:
            ms = self._time_to_use(args)
            return max(1, min(self.options.simulations * 10, ms // 10))
        return self.options.simulations

    def _time_to_use(self, args: list[str]) -> int:
        remaining_key = "wtime" if self.board.turn == 0 else "btime"
        inc_key = "winc" if self.board.turn == 0 else "binc"
        opponent_key = "btime" if self.board.turn == 0 else "wtime"
        remaining = int(args[args.index(remaining_key) + 1]) if remaining_key in args else 1000
        increment = int(args[args.index(inc_key) + 1]) if inc_key in args else 0
        opponent_remaining = int(args[args.index(opponent_key) + 1]) if opponent_key in args else remaining
        use = int(remaining / 40 + increment * 0.8)
        if remaining > opponent_remaining:
            use = int(use * 1.2)
        cap = max(1, int(remaining * 0.10))
        use = min(use, cap)
        if remaining < opponent_remaining:
            return max(500, use)
        return max(100, use)

    def _load_checkpoint(self) -> None:
        if not self.options.checkpoint:
            self.evaluator = UniformEvaluator()
        else:
            from .model import load_model

            model = load_model(self.options.checkpoint, self.options.device)
            self.evaluator = NetworkEvaluator(model, self.options.device)
        self.mcts = MCTS(
            self.evaluator,
            simulations=self.options.simulations,
            c_puct=self.options.cpuct,
            batch_size=32,
            add_noise=False,
            resign_threshold=-1.0,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--simulations", type=int, default=200)
    args, _ = parser.parse_known_args(argv)
    UCIEngine(UCIOptions(args.simulations, checkpoint=args.checkpoint, device=args.device)).loop()


if __name__ == "__main__":  # pragma: no cover
    main()
