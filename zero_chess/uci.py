"""Universal Chess Interface (UCI) entry point supporting subtree reuse."""

from __future__ import annotations

import argparse
import shlex
import sys
import threading
from dataclasses import dataclass

from .board import Board
from .constants import WHITE
from .mcts import MCTS, NetworkEvaluator, SearchResult, UniformEvaluator
from .move import Move


@dataclass(slots=True)
class UCIOptions:
    """UCI engine options: simulations, CPuct, checkpoint path, and device."""

    simulations: int = 200
    cpuct: float = 1.5
    checkpoint: str | None = None
    device: str = "cpu"


class UCIEngine:
    """Manages the standard Universal Chess Interface protocol for zero-latency GUI play."""

    def __init__(self, options: UCIOptions | None = None) -> None:
        self.options = options or UCIOptions()
        self.board = Board()
        self.position_history: list[Board] = []
        self.played_moves: list[str] = []
        self.evaluator = UniformEvaluator()
        self.mcts = MCTS(
            self.evaluator, simulations=self.options.simulations, c_puct=self.options.cpuct, add_noise=False
        )
        self._search_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._best_result: SearchResult | None = None
        self._bestmove_emitted = False
        self._search_token = 0
        self._position_seen = False
        self._searched = False
        self._state_lock = threading.RLock()
        self._base_fen = self.board.fen()
        self._load_checkpoint()

    def loop(self) -> None:
        """Standard input loop parsing UCI commands asynchronously."""
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
                self._cmd_stop(emit=False)
                with self._state_lock:
                    self.board = Board()
                    self.position_history.clear()
                    self.played_moves.clear()
                    self._base_fen = self.board.fen()
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
        self._cmd_stop(emit=False)
        with self._state_lock:
            self._cmd_setoption_locked(args)

    def _cmd_setoption_locked(self, args: list[str]) -> None:
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
        self._cmd_stop(emit=False)
        with self._state_lock:
            self._cmd_position_locked(args)

    def _cmd_position_locked(self, args: list[str]) -> None:
        """Parse position commands, utilizing fast subtree reuse where possible."""
        if not args:
            return
        idx = 0
        base_board = Board()
        if args[idx] == "startpos":
            base_board = Board()
            idx += 1
        elif args[idx] == "fen":
            fen_fields = []
            idx += 1
            while idx < len(args) and args[idx] != "moves":
                fen_fields.append(args[idx])
                idx += 1
            base_board = Board.from_fen(" ".join(fen_fields))
        else:
            raise ValueError("position requires startpos or fen")

        new_moves = args[idx + 1 :] if (idx < len(args) and args[idx] == "moves") else []
        base_fen = base_board.fen()

        # Validate the complete command on an isolated board first. A malformed
        # later move must never leave the live UCI position partially applied.
        candidate_board = base_board
        candidate_history: list[Board] = []
        for move_str in new_moves:
            resolved_move = self._resolve_legal_move(candidate_board, move_str)
            candidate_history.insert(0, candidate_board.copy())
            candidate_board._push_unchecked(resolved_move)

        # Check if the new state is an incremental extension of our existing active board
        if (
            self.played_moves
            and base_fen == self._base_fen
            and len(new_moves) >= len(self.played_moves)
            and new_moves[: len(self.played_moves)] == self.played_moves
        ):
            # Incremental Update (Tree Reuse Path)
            added_moves = new_moves[len(self.played_moves) :]
            for move_str in added_moves:
                resolved_move = self._resolve_legal_move(self.board, move_str)
                self._record_and_push(resolved_move)
                self.mcts.advance_to(resolved_move, history=self.position_history)
            self.played_moves = new_moves
        else:
            # Full Reset
            self.board = candidate_board
            self.position_history = candidate_history
            self._base_fen = base_fen
            self.mcts.reset()
            self.played_moves = new_moves

    @staticmethod
    def _resolve_legal_move(board: Board, move_str: str) -> Move:
        raw_move = Move.from_uci(move_str)
        for legal in board.legal_moves():
            if (
                legal.from_sq == raw_move.from_sq
                and legal.to_sq == raw_move.to_sq
                and legal.promotion == raw_move.promotion
            ):
                return legal
        raise ValueError(f"illegal move {move_str!r} in position {board.fen()}")

    def _cmd_go(self, args: list[str]) -> None:
        self._searched = True
        if "infinite" in args:
            self._start_infinite_search()
            return
        self._start_finite_search(args)

    def _start_finite_search(self, args: list[str]) -> None:
        if self._search_thread and self._search_thread.is_alive():
            self._cmd_stop(emit=False)
        self._search_token += 1
        token = self._search_token
        stop_event = threading.Event()
        self._stop_event = stop_event
        self._best_result = None
        self._bestmove_emitted = False

        def worker() -> None:
            with self._state_lock:
                if "movetime" in args:
                    ms = int(args[args.index("movetime") + 1])
                    result = self.mcts.search_time(
                        self.board,
                        ms,
                        temperature=0.0,
                        add_noise=False,
                        history=self.position_history,
                        stop_event=stop_event,
                    )
                elif "wtime" in args or "btime" in args:
                    ms = self._time_to_use(args)
                    result = self.mcts.search_time(
                        self.board,
                        ms,
                        temperature=0.0,
                        add_noise=False,
                        history=self.position_history,
                        stop_event=stop_event,
                    )
                else:
                    simulations = self._simulations_for_go(args)
                    result = self.mcts.search(
                        self.board,
                        num_simulations=simulations,
                        temperature=0.0,
                        add_noise=False,
                        history=self.position_history,
                        stop_event=stop_event,
                    )
                if token != self._search_token:
                    return
                self._best_result = result
                if not stop_event.is_set() and not self._bestmove_emitted:
                    self._emit_bestmove(result)
                    self._bestmove_emitted = True

        self._search_thread = threading.Thread(target=worker, daemon=True)
        self._search_thread.start()

    def _start_infinite_search(self) -> None:
        if self._search_thread and self._search_thread.is_alive():
            self._cmd_stop(emit=False)
            if self._search_thread.is_alive():
                return
        self._search_token += 1
        token = self._search_token
        stop_event = threading.Event()
        self._stop_event = stop_event
        self._best_result = None
        self._bestmove_emitted = False

        def worker() -> None:
            while not stop_event.is_set():
                with self._state_lock:
                    if stop_event.is_set() or token != self._search_token:
                        break
                    self._best_result = self.mcts.search(
                        self.board,
                        num_simulations=self.mcts.batch_size,
                        temperature=0.0,
                        add_noise=False,
                        stop_event=stop_event,
                        history=self.position_history,
                    )

        self._search_thread = threading.Thread(target=worker, daemon=True)
        self._search_thread.start()

    def _cmd_stop(self, emit: bool = True) -> None:
        self._stop_event.set()
        self._search_token += 1
        if self._search_thread and self._search_thread.is_alive():
            self._search_thread.join(timeout=1.0)
        if emit and not self._bestmove_emitted:
            self._emit_bestmove(self._best_result)
            self._bestmove_emitted = True

    def _emit_bestmove(self, result: SearchResult | None) -> None:
        if result is None or result.move is None:
            legal = self.board.legal_moves()
            best = legal[0].uci() if legal else "0000"
            nodes = 0
            score_cp = 0
        else:
            best = result.move.uci()
            nodes = sum(result.visits.values())
            # ZERO-X values are zero-sum: 0.0 is a draw and ±1.0 are decisive.
            score_cp = int(result.root_q_with_contempt * 100)

        print(f"info depth 1 nodes {nodes} score cp {score_cp}", flush=True)
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
        remaining_key = "wtime" if self.board.turn == WHITE else "btime"
        inc_key = "winc" if self.board.turn == WHITE else "binc"
        opponent_key = "btime" if self.board.turn == WHITE else "wtime"
        remaining = int(args[args.index(remaining_key) + 1]) if remaining_key in args else 1000
        increment = int(args[args.index(inc_key) + 1]) if inc_key in args else 0
        opponent_remaining = int(args[args.index(opponent_key) + 1]) if opponent_key in args else remaining
        use = int(remaining / 40 + increment * 0.8)
        if remaining > opponent_remaining:
            use = int(use * 1.2)
        if remaining < opponent_remaining:
            use = max(50, int(remaining * 0.15))
        else:
            use = max(100, use)
        safety_margin = max(50, int(remaining * 0.10))
        max_alloc = max(1, remaining - safety_margin)
        return max(1, min(use, max_alloc, remaining))

    def _load_checkpoint(self) -> None:
        with self._state_lock:
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

    def _record_and_push(self, move: Move) -> None:
        """Keep full newest-first history; encoding truncates it to seven planes."""
        self.position_history.insert(0, self.board.copy())
        self.board._push_unchecked(move)

    def _record_and_push_uci(self, text: str) -> None:
        """Resolve and play UCI while preserving history for neural inputs."""
        raw_move = Move.from_uci(text)
        for legal in self.board.legal_moves():
            if (
                legal.from_sq == raw_move.from_sq
                and legal.to_sq == raw_move.to_sq
                and legal.promotion == raw_move.promotion
            ):
                self._record_and_push(legal)
                return
        raise ValueError(f"illegal move {text!r} in position {self.board.fen()}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: start the UCI engine loop on stdin/stdout."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--simulations", type=int, default=200)
    args, _ = parser.parse_known_args(argv)
    UCIEngine(UCIOptions(args.simulations, checkpoint=args.checkpoint, device=args.device)).loop()


if __name__ == "__main__":  # pragma: no cover
    main()
