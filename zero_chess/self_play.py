"""Highly optimized parallel and persistent self-play game generator."""

from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import random
import subprocess
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import multiprocessing

# CONFIGURATION & DATA STRUCTURES

from .board import Board
from .constants import (
    BLACK, BK, BQ, WHITE, WK, WQ,
    file_of, parse_square, rank_of, square, square_name
)
from .elo import DEFAULT_ELO, update_rating_from_result
from .encoding import terminal_wdl
from .mcts import MCTS, NetworkEvaluator, UniformEvaluator
from .move import Move
from .pgn import export_pgn
from .replay import Experience, PrioritizedReplayBuffer
from .targets import (
    AGGRESSION_WEIGHT, MOMENTUM_REWARD, PANIC_PENALTY,
    opponent_value,
)


def _ensure_importable() -> None:
    """Ensure zero_chess is importable in Windows spawn child processes."""
    import sys
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

@dataclass(slots=True)
class SelfPlayConfig:
    """Configuration for self-play game generation: simulations, batch size, and opening randomness."""
    simulations: int = 200
    batch_size: int = 24
    max_plies: int = 512
    temperature_moves: int = 30
    opening_random_plies: int = 6
    opening_random_prob: float = 0.5
    resign_value: float = -0.95
    disable_resign: bool = False
    generation: int = 0
    reuse_tree: bool = True

@dataclass(slots=True)
class PositionRecord:
    """A single position recorded during self-play for later training target generation."""
    fen: str
    policy: dict[str, float]
    turn: int
    root_value: float
    aggression_score: float
    momentum_reward: float
    panic_penalty: float


def _make_mcts(evaluator, config: SelfPlayConfig) -> MCTS:
    """Build an MCTS searcher configured from self-play settings."""
    resign_threshold = -1.0 if config.disable_resign else config.resign_value
    return MCTS(
        evaluator,
        c_puct=1.5 if config.generation < 200 else 1.0,
        batch_size=config.batch_size,
        simulations=config.simulations,
        resign_threshold=resign_threshold,
    )

# CORE GAME LOOP & EXPERIENCE GENERATION 

def play_game(
    mcts: MCTS,
    config: SelfPlayConfig | None = None,
    rng: random.Random | None = None,
) -> tuple[str, list[Experience], list[str], str, dict]:
    """Play a full self-play game and return (result, experiences, SAN moves, reason, metadata)."""
    config = config or SelfPlayConfig()
    rng = rng or random.Random()
    board = Board()
    records: list[PositionRecord] = []
    sans: list[str] = []
    adjudication: list[float] = []
    start_time = time.monotonic()

    for ply in range(config.max_plies):
        result = board.outcome()
        if result is not None:
            if board.is_checkmate():
                reason = "checkmate"
            elif board.is_stalemate():
                reason = "stalemate"
            elif board.has_insufficient_material():
                reason = "draw_material"
            else:
                reason = "draw_repetition"
            
            meta = _compile_game_metadata(board, sans, start_time)
            return result, _finalize(records, result, reason, rng), sans, reason, meta

        temperature = 1.0 if ply < config.temperature_moves else 0.0
        search = mcts.search(
            board,
            num_simulations=config.simulations,
            temperature=temperature,
            add_noise=True,
            generation=config.generation,
        )
        if search.resigned and not config.disable_resign:
            result = "0-1" if board.turn == WHITE else "1-0"
            meta = _compile_game_metadata(board, sans, start_time)
            return result, _finalize(records, result, "resignation", rng), sans, "resignation", meta

        legal = board.legal_moves()
        if ply < config.opening_random_plies and rng.random() < config.opening_random_prob and legal:
            move = rng.choice(legal)
        else:
            move = search.move or rng.choice(legal)

        policy = search.policy
        root_value = search.root.q
        records.append(
            PositionRecord(
                fen=board.fen(),
                policy={move_.uci(): prob for move_, prob in policy.items()},
                turn=board.turn,
                root_value=root_value,
                aggression_score=_aggression_score(board),
                momentum_reward=_momentum_reward(board, move),
                panic_penalty=_panic_penalty(board, move),
            )
        )

        white_q = root_value if board.turn == WHITE else opponent_value(root_value)
        adjudication.append(white_q)
        if len(adjudication) >= 20 and _adjudicate(adjudication[-20:]):
            avg = sum(adjudication[-20:]) / 20.0
            winning_side_is_white = avg > -1.0
            result = "1-0" if winning_side_is_white else "0-1"
            meta = _compile_game_metadata(board, sans, start_time)
            return result, _finalize(records, result, "adjudication", rng), sans, "adjudication", meta

        sans.append(board.san(move))
        board.push(move)

        if config.reuse_tree:
            mcts.advance_to(move)
        else:
            mcts.reset()

    meta = _compile_game_metadata(board, sans, start_time)
    return "1/2-1/2", _finalize(records, "1/2-1/2", "max_plies", rng), sans, "max_plies", meta

def _compile_game_metadata(board: Board, sans: list[str], start_time: float) -> dict:
    """Calculates active game metrics and positional attributes at termination."""
    values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "p": -1, "n": -3, "b": -3, "r": -5, "q": -9}
    material_delta = sum(values.get(p, 0) for p in board.squares if p != ".")
    opening_str = " ".join(sans[:6]) if len(sans) >= 6 else " ".join(sans)
    return {
        "duration": time.monotonic() - start_time,
        "material_delta": material_delta,
        "opening": opening_str or "Standard Open"
    }

def _finalize(
    records: list[PositionRecord],
    result: str,
    reason_or_rng: str | random.Random | None = "checkmate",
    rng: random.Random | None = None,
    augment: bool = False,
) -> list[Experience]:
    if isinstance(reason_or_rng, random.Random):
        rng = reason_or_rng
        reason = "checkmate"
    elif reason_or_rng is None:
        reason = "checkmate"
    else:
        reason = reason_or_rng

    experiences = []
    
    # Calculate custom rewards mapping directly to the new scale
    if result == "1/2-1/2":
        if reason == "stalemate":
            rewards = (-10.0, -10.0)
        elif reason == "max_plies":
            rewards = (-20.0, -20.0)
        else:
            rewards = (-1.0, -1.0)
    elif result == "1-0":
        if reason == "resignation":
            rewards = (0.0, -30.0)
        else:
            rewards = (1.0, -3.0)
    else:
        if reason == "resignation":
            rewards = (-30.0, 0.0)
        else:
            rewards = (-3.0, 1.0)
            
    for idx, record in enumerate(records):
        terminal_reward = rewards[0] if record.turn == WHITE else rewards[1]
        bootstrap = _bootstrap_value(records, idx)
        reward_bonus = (
            AGGRESSION_WEIGHT * record.aggression_score
            + record.momentum_reward
            + record.panic_penalty
        )
        exp = Experience(
            fen=record.fen,
            policy=record.policy,
            value=terminal_reward,
            td_value=bootstrap,
            wdl=terminal_wdl(terminal_reward),
            priority=abs(record.root_value - (terminal_reward + reward_bonus)) + 1e-3,
            value_prediction=record.root_value,
            reward_bonus=reward_bonus,
            aggression_score=record.aggression_score,
            momentum_reward=record.momentum_reward,
            panic_penalty=record.panic_penalty,
        )
        experiences.append(exp)
        if augment and rng is not None and rng.random() < 0.5:
            experiences.append(_flip_experience_horizontal(exp))
    return experiences

def _bootstrap_value(records: list[PositionRecord], idx: int, plies: int = 5) -> float:
    if not records:
        return -1.0
    j = min(len(records) - 1, idx + plies)
    current_turn = records[idx].turn
    later_turn = records[j].turn
    later_value = records[j].root_value
    return later_value if later_turn == current_turn else opponent_value(later_value)

def _flip_square_name(name: str) -> str:
    sq = parse_square(name)
    return square_name(square(7 - file_of(sq), rank_of(sq)))

def _flip_uci_horizontal(uci: str) -> str:
    if len(uci) >= 4:
        promo = uci[4:]
        return _flip_square_name(uci[:2]) + _flip_square_name(uci[2:4]) + promo
    return uci

def _flip_board_horizontal(board: Board) -> Board:
    squares = ["."] * 64
    for sq, piece in enumerate(board.squares):
        squares[square(7 - file_of(sq), rank_of(sq))] = piece
    rights = 0
    if board.castling_rights & WK:
        rights |= WQ
    if board.castling_rights & WQ:
        rights |= WK
    if board.castling_rights & BK:
        rights |= BQ
    if board.castling_rights & BQ:
        rights |= BK
    ep = None if board.ep_square is None else square(7 - file_of(board.ep_square), rank_of(board.ep_square))
    return Board(squares, board.turn, rights, ep, board.halfmove_clock, board.fullmove_number)

def _flip_experience_horizontal(exp: Experience) -> Experience:
    board = _flip_board_horizontal(Board.from_fen(exp.fen))
    return Experience(
        fen=board.fen(),
        policy={_flip_uci_horizontal(uci): prob for uci, prob in exp.policy.items()},
        value=exp.value,
        td_value=exp.td_value,
        wdl=exp.wdl,
        priority=exp.priority,
        value_prediction=exp.value_prediction,
        importance_weight=exp.importance_weight,
        reward_bonus=exp.reward_bonus,
        aggression_score=exp.aggression_score,
        momentum_reward=exp.momentum_reward,
        panic_penalty=exp.panic_penalty,
    )

def generate_parallel_games(
    evaluator,
    games: int = 2,
    config: SelfPlayConfig | None = None,
    rng_seed: int | None = None,
) -> list[tuple[str, list[Experience], list[str], str, dict]]:
    """Generate ``games`` self-play games in parallel threads using a single evaluator."""
    config = config or SelfPlayConfig()
    results: list[tuple[str, list[Experience], list[str], str, dict] | None] = [None] * games

    wrapped_evaluator = evaluator
    shared_evaluator = None
    if games > 1 and isinstance(evaluator, NetworkEvaluator):
        from .mcts import SharedBatchEvaluator
        shared_evaluator = SharedBatchEvaluator(
            evaluator.model,
            device=evaluator.device,
            max_batch_size=max(128, config.batch_size * games),
        )
        wrapped_evaluator = shared_evaluator

    def worker(index: int) -> None:
        rng = random.Random(None if rng_seed is None else rng_seed + index)
        mcts = _make_mcts(wrapped_evaluator, config)
        results[index] = play_game(mcts, config, rng)
        mcts.reset()
        del mcts

    try:
        threads = [threading.Thread(target=worker, args=(idx,), daemon=True) for idx in range(games)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        if shared_evaluator is not None:
            shared_evaluator.close()

    return [result for result in results if result is not None]

class QueueEvaluatorProxy:
    """MCTS evaluator proxy: encodes boards to tensors (CPU) then sends to GPU evaluator."""

    def __init__(self, worker_id: int, request_queue, response_queue, stop_event=None) -> None:
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.stop_event = stop_event
        self._next_request_id = 0

    def evaluate_batch(self, boards: list[Board]):
        legal_moves = [b.legal_moves() for b in boards]
        fens = [b.fen() for b in boards]
        legal_ucis = [[m.uci() for m in legal] for legal in legal_moves]

        self._next_request_id += 1
        request_id = self._next_request_id
        # Send compact FEN + legal-move descriptors instead of full tensors to keep IPC tiny.
        self.request_queue.put(("eval", self.worker_id, request_id, (fens, legal_ucis)))
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise RuntimeError("Evaluation aborted: stop event is set")
            try:
                response_id, data, error = self.response_queue.get(timeout=1.0)
            except queue.Empty:
                if os.getppid() == 1:
                    raise RuntimeError("Evaluation aborted: parent process died")
                # On Windows, getppid() doesn't return 1 for orphans.
                # The stop_event check above is the primary signal; this is a Unix fallback.
                continue
            if response_id != request_id:
                continue
            if error is not None:
                raise RuntimeError(error)

            raw_values, raw_uncertainties, legal_probs = data
            results = []
            for i, legal in enumerate(legal_moves):
                probs = legal_probs[i]
                results.append(({m: float(p) for m, p in zip(legal, probs)}, float(raw_values[i]), float(raw_uncertainties[i])))
            return results

def generate_multiprocess_games(
    model,
    device: str = "cuda",
    games: int = 2,
    config: SelfPlayConfig | None = None,
    rng_seed: int | None = None,
    gpu_batch_size: int = 64,
    max_wait_ms: float = 20.0,
) -> list[tuple[str, list[Experience], list[str], str, dict]]:
    """Generate self-play with CPU worker processes feeding one CUDA evaluator process."""
    config = config or SelfPlayConfig()
    ctx = multiprocessing.get_context("spawn")
    request_queue = ctx.Queue()
    response_queues = [ctx.Queue() for _ in range(games)]
    result_queue = ctx.Queue()
    active_workers = ctx.Array('b', [True] * games, lock=False)
    model_config, state_dict = _model_payload_for_process(model)

    evaluator = ctx.Process(
        target=_gpu_evaluator_process_main,
        args=(model_config, state_dict, device, request_queue, response_queues, gpu_batch_size, max_wait_ms, active_workers),
        daemon=True,
    )
    evaluator.start()

    workers = []
    for idx in range(games):
        seed = None if rng_seed is None else rng_seed + idx
        process = ctx.Process(
            target=_self_play_process_main,
            args=(idx, request_queue, response_queues[idx], result_queue, config, seed, active_workers),
            daemon=True,
        )
        process.start()
        workers.append(process)

    results: list[tuple[str, list[Experience], list[str], str, dict] | None] = [None] * games
    received = 0
    while received < games:
        try:
            idx, payload, error = result_queue.get(timeout=1.0)
        except queue.Empty:
            # Check worker process health and re-spawn if dead
            for i, process in enumerate(workers):
                if not process.is_alive() and results[i] is None:
                    print(f"[WARNING] Self-play worker process {i} died silently. Re-spawning.", flush=True)
                    seed = None if rng_seed is None else rng_seed + i
                    new_process = ctx.Process(
                        target=_self_play_process_main,
                        args=(i, request_queue, response_queues[i], result_queue, config, seed, active_workers),
                        daemon=True,
                    )
                    new_process.start()
                    workers[i] = new_process
            continue
        received += 1
        if error is None:
            results[idx] = payload
        else:
            print(f"[ERROR] Self-play worker {idx} failed:\n{error}", flush=True)

    for process in workers:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
    request_queue.put(("stop",))
    evaluator.join(timeout=5.0)
    if evaluator.is_alive():
        evaluator.terminate()
        evaluator.join(timeout=2.0)
        if evaluator.is_alive():
            evaluator.kill()
    try:
        request_queue.close()
        result_queue.close()
        for q in response_queues:
            q.close()
    except Exception:
        pass
    return [result for result in results if result is not None]

@dataclass(slots=True)
class CudaTrainingRuntime:
    """Handle for a persistent CUDA training runtime; use as a context manager or call stop()."""
    stop_event: object
    processes: list[object]
    resources: list[object]
    generation_value: object
    games_completed: object

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def stop(self, timeout: float = 10.0) -> None:
        self.stop_event.set()
        _terminate_delay = 1.0
        
        # Terminate processes FIRST to break any blocked Queue.put() calls.
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        
        for process in self.processes:
            process.join(timeout=_terminate_delay)
            if process.is_alive():
                process.kill()
                process.join(timeout=_terminate_delay)
        
        # Safely close and drain all multiprocessing queues after process termination.
        def _close_and_drain(q) -> None:
            if hasattr(q, "close") and hasattr(q, "join_thread"):
                try:
                    q.close()
                except Exception:
                    pass
                while True:
                    try:
                        q.get_nowait()
                    except Exception:
                        break
                try:
                    q.join_thread()
                except Exception:
                    pass

        for res in self.resources:
            if isinstance(res, list):
                for item in res:
                    _close_and_drain(item)
            else:
                _close_and_drain(res)

def start_persistent_cuda_training(
    model,
    train_config,
    *,
    games: int = 2,
    self_play_config: SelfPlayConfig | None = None,
    device: str = "cuda",
    gpu_batch_size: int = 2048,
    eval_wait_ms: float = 5.0,
    cuda_memory_fraction: float = 0.90,
    gpu_cooldown_ms: float = 0.0,
    compile_model: bool = False,
    updates_per_game: float = 8.0,
    iteration: int = 0,
    train_updates: int = 0,
    elo: float = DEFAULT_ELO,
    replay_path: str = "data/replay.pkl",
    cold_replay_path: str = "data/replay_cold.sqlite3",
    checkpoint_dir: str = "checkpoints",
    training_log_path: str = "logs/training.log",
    monitor_log_path: str = "logs/utilization.log",
) -> CudaTrainingRuntime:
    """Start persistent self-play optimized for Blackwell 24GB VRAM and 128GB RAM."""
    config = self_play_config or SelfPlayConfig()
    ctx = multiprocessing.get_context("spawn")
    request_queue = ctx.Queue(maxsize=128)
    response_queues = [ctx.Queue(maxsize=16) for _ in range(games)]
    game_queue = ctx.Queue(maxsize=128)
    stop_event = ctx.Event()
    weights_updated = ctx.Event()
    state_lock = ctx.Lock()
    generation_value = ctx.Value("i", int(iteration))
    games_completed = ctx.Value("q", 0)
    positions_evaluated = ctx.Value("q", 0)
    eval_batches = ctx.Value("q", 0)
    batch_positions = ctx.Value("q", 0)
    active_workers = ctx.Array('b', [True] * games, lock=False)
    model_config, shared_state = _model_payload_for_process(model)
    train_config_dict = asdict(train_config)

    # Split VRAM budget safely between processes (e.g., if input is 0.90):
    # Evaluator: 0.90 * 0.25 = 22.5% VRAM (~5.4 GB) -> Plenty for batch size 256
    # Trainer:   0.90 * 0.70 = 63.0% VRAM (~15.0 GB) -> Remaining for heavy backpropagation
    eval_memory_fraction = cuda_memory_fraction * 0.25
    train_memory_fraction = cuda_memory_fraction * 0.70

    evaluator = ctx.Process(
        target=_persistent_gpu_evaluator_process_main,
        args=(
            model_config,
            shared_state,
            state_lock,
            weights_updated,
            stop_event,
            device,
            request_queue,
            response_queues,
            gpu_batch_size,
            eval_wait_ms,
            eval_memory_fraction,  # <-- Passes optimized 22.5%
            gpu_cooldown_ms,
            compile_model,
            positions_evaluated,
            eval_batches,
            batch_positions,
            active_workers,
        ),
        daemon=False,
    )
    evaluator.start()

    trainer = ctx.Process(
        target=_trainer_process_main,
        args=(
            model_config,
            shared_state,
            state_lock,
            weights_updated,
            stop_event,
            game_queue,
            generation_value,
            train_config_dict,
            device,
            train_memory_fraction,  # <-- Passes optimized 63%
            updates_per_game,
            iteration,
            train_updates,
            elo,
            replay_path,
            cold_replay_path,
            checkpoint_dir,
            training_log_path,
            games,
        ),
        daemon=False,
    )
    trainer.start()

    respawn_counters = [0] * games
    workers = []
    for worker_id in range(games):
        process = ctx.Process(
            target=_persistent_self_play_process_main,
            args=(
                worker_id,
                request_queue,
                response_queues[worker_id],
                game_queue,
                stop_event,
                generation_value,
                games_completed,
                config,
                iteration * 100_000 + worker_id,
                active_workers,
            ),
            daemon=False,
        )
        process.start()
        workers.append(process)

    monitor = ctx.Process(
        target=_utilization_monitor_process_main,
        args=(
            stop_event,
            games_completed,
            positions_evaluated,
            eval_batches,
            batch_positions,
            gpu_batch_size,
            monitor_log_path,
        ),
        daemon=False,
    )
    monitor.start()

    # Collect and keep strong references to all IPC resources to prevent GC unlinking
    resources = [
        request_queue,
        response_queues,
        game_queue,
        stop_event,
        weights_updated,
        state_lock,
        generation_value,
        games_completed,
        positions_evaluated,
        eval_batches,
        batch_positions,
        active_workers,
    ]

    runtime = CudaTrainingRuntime(stop_event, [*workers, evaluator, trainer, monitor], resources, generation_value, games_completed)

    def monitor_workers():
        while not stop_event.is_set():
            time.sleep(2.0)
            for i in range(games):
                if stop_event.is_set():
                    break
                process = workers[i]
                if not process.is_alive():
                    print(f"[WARNING] Persistent worker process {i} died silently. Re-spawning.", flush=True)
                    respawn_counters[i] += 1
                    seed = int(generation_value.value) * 100_000 + i + (respawn_counters[i] * 10_000)
                    new_process = ctx.Process(
                        target=_persistent_self_play_process_main,
                        args=(
                            i,
                            request_queue,
                            response_queues[i],
                            game_queue,
                            stop_event,
                            generation_value,
                            games_completed,
                            config,
                            seed,
                            active_workers,
                        ),
                        daemon=False,
                    )
                    new_process.start()
                    workers[i] = new_process
                    try:
                        idx = runtime.processes.index(process)
                        runtime.processes[idx] = new_process
                    except ValueError:
                        runtime.processes.append(new_process)
            
            # Check evaluator and trainer health
            if not stop_event.is_set():
                if not evaluator.is_alive():
                    print("[WARNING] Evaluator process died silently! Shutting down training session.", flush=True)
                    stop_event.set()
                elif not trainer.is_alive():
                    print("[WARNING] Trainer process died silently! Shutting down training session.", flush=True)
                    stop_event.set()

    monitor_thread = threading.Thread(target=monitor_workers, name="zero-runtime-monitor", daemon=True)
    monitor_thread.start()

    return runtime

# WORKER PROCESS ENTRYPOINTS

def _persistent_self_play_process_main(
    worker_id: int,
    request_queue,
    response_queue,
    game_queue,
    stop_event,
    generation_value,
    games_completed,
    base_config: SelfPlayConfig,
    rng_seed: int,
    active_workers,
) -> None:
    _ensure_importable()

    # Win32 Process Priority Optimization
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception:
        pass

    active_workers[worker_id] = True
    try:
        rng = random.Random(rng_seed)
        evaluator = QueueEvaluatorProxy(worker_id, request_queue, response_queue, stop_event)
        while not stop_event.is_set():
            # Memory Guard: Halt and flush if free system RAM falls below 512MB
            _, _, ram_avail, _, _ = _query_memory_utilization()
            if ram_avail < 512:
                gc.collect()
                time.sleep(5.0)
                continue

            generation = int(generation_value.value)
            config = replace(base_config, generation=generation)
            mcts = _make_mcts(evaluator, config)
            try:
                game_result = play_game(mcts, config, rng)
                game_queue.put((worker_id, game_result, None))
                mcts.reset()
                del mcts
                with games_completed.get_lock():
                    games_completed.value += 1
            except Exception:
                game_queue.put((worker_id, None, traceback.format_exc()))
    finally:
        active_workers[worker_id] = False

def _persistent_gpu_evaluator_process_main(
    model_config: dict,
    shared_state: dict,
    state_lock,
    weights_updated,
    stop_event,
    device: str,
    request_queue,
    response_queues,
    gpu_batch_size: int,
    eval_wait_ms: float,
    cuda_memory_fraction: float,
    gpu_cooldown_ms: float,
    compile_model: bool,
    positions_evaluated,
    eval_batches,
    batch_positions,
    active_workers,
) -> None:
    _ensure_importable()
    import torch
    from .encoding import INPUT_CHANNELS, POLICY_SIZE
    from .model import ModelConfig, ZeroNet

    _configure_cuda_process(device, torch, cuda_memory_fraction)
    model = ZeroNet(ModelConfig(**model_config)).to(device)
    with state_lock:
        model.load_state_dict(shared_state)
    model.eval()
    if compile_model:
        try:
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        except AttributeError:
            pass
        compiled_model = torch.compile(model, mode="reduce-overhead")
    else:
        compiled_model = model

    # Pre-allocate static padding tensors to eliminate repeated allocations on every eval step.
    static_tensor_pad = torch.zeros((gpu_batch_size - 1, INPUT_CHANNELS, 8, 8), dtype=torch.float32)
    static_mask_pad = torch.zeros((gpu_batch_size - 1, POLICY_SIZE), dtype=torch.float32)
    wait_seconds = eval_wait_ms / 1000.0

    while not stop_event.is_set():
        if weights_updated.is_set():
            with state_lock:
                model.load_state_dict(shared_state)
                weights_updated.clear()
            model.eval()

        requests = _collect_eval_requests_nonblocking(request_queue, gpu_batch_size, wait_seconds, stop_event)
        if requests is None:
            return  # Clean stop
        if not requests:
            continue  # Idle loop back

        flat_fens = []
        flat_legal_ucis = []
        for _, _, _, data in requests:
            fens, legal_ucis = data
            flat_fens.extend(fens)
            flat_legal_ucis.extend(legal_ucis)

        total_positions = len(flat_fens)
        if total_positions == 0:
            continue

        with positions_evaluated.get_lock():
            positions_evaluated.value += total_positions
        with eval_batches.get_lock():
            eval_batches.value += 1
        with batch_positions.get_lock():
            batch_positions.value += total_positions

        try:
            from .board import Board
            from .encoding import INPUT_CHANNELS, POLICY_SIZE, encode_boards, move_to_policy_index
            from .move import Move

            boards = [Board.from_fen(fen) for fen in flat_fens]
            tensors = encode_boards(boards, device="cpu")
            masks = torch.stack([_encode_move_mask_from_ucis(b, legal, device="cpu") for b, legal in zip(boards, flat_legal_ucis)])

            # STATIC BATCH PADDING FIX
            # Use pre-allocated static padding tensors to eliminate repeated allocations.
            actual_size = tensors.shape[0]
            if actual_size < gpu_batch_size:
                padding_size = gpu_batch_size - actual_size
                tensors = torch.cat([tensors, static_tensor_pad[:padding_size]], dim=0)
                masks = torch.cat([masks, static_mask_pad[:padding_size]], dim=0)

            is_cuda = device == "cuda" and torch.cuda.is_available()
            if is_cuda:
                tensors = tensors.pin_memory().to(device, non_blocking=True)
                masks = masks.pin_memory().to(device, non_blocking=True)
            else:
                tensors = tensors.to(device)
                masks = masks.to(device)
            amp_dtype = torch.bfloat16 if is_cuda and torch.cuda.is_bf16_supported() else torch.float16
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=is_cuda):
                    out = compiled_model(tensors, masks, return_dict=True)

            # Slice the outputs back to their actual unpadded sizes
            raw_values = out["value"][:actual_size].squeeze(-1).detach().cpu().tolist()
            raw_uncertainties = out["uncertainty"][:actual_size].detach().cpu().tolist()
            raw_policies = out["policy"][:actual_size].detach().cpu()

            legal_probs = []
            for batch_idx, (board, legal) in enumerate(zip(boards, flat_legal_ucis)):
                indices = torch.tensor([move_to_policy_index(board, Move.from_uci(uci)) for uci in legal], dtype=torch.long)
                legal_probs.append(raw_policies[batch_idx][indices].tolist())
            error = None
        except Exception as exc:
            if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower() or "timeout" in str(exc).lower():
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[WARNING] Persistent GPU evaluator forward pass failed: {exc}. Flushing CUDA cache and falling back to uniform evaluation.", flush=True)
                raw_values = [0.0] * len(flat_fens)
                raw_uncertainties = [1.0] * len(flat_fens)
                legal_probs = []
                for legal in flat_legal_ucis:
                    if not legal:
                        legal_probs.append([])
                    else:
                        prob = 1.0 / len(legal)
                        legal_probs.append([prob] * len(legal))
                error = None
            else:
                raw_values = raw_uncertainties = []
                legal_probs = []
                error = traceback.format_exc()

        offset = 0
        for _, worker_id, request_id, data in requests:
            batch_size = len(data[0])
            if error:
                response_queues[worker_id].put((request_id, (None,), error))
            else:
                response_queues[worker_id].put((
                    request_id,
                    (raw_values[offset:offset + batch_size],
                     raw_uncertainties[offset:offset + batch_size],
                     legal_probs[offset:offset + batch_size]),
                    None,
                ))
            offset += batch_size

        # Fulfill defined GPU cooldown sleep parameter to avoid thermal limits
        if gpu_cooldown_ms > 0.0:
            time.sleep(gpu_cooldown_ms / 1000.0)

def _encode_move_mask_from_ucis(board: Board, legal_ucis: list[str], device: str | None = None):
    """Build a float legal-move mask from pre-computed UCI strings (no re-generation)."""
    import torch
    from .encoding import POLICY_SIZE, move_to_policy_index
    from .move import Move

    mask = torch.zeros(POLICY_SIZE, dtype=torch.float32, device=device)
    for uci in legal_ucis:
        mask[move_to_policy_index(board, Move.from_uci(uci))] = 1.0
    return mask

def _collect_eval_requests_nonblocking(request_queue, gpu_batch_size: int, wait_seconds: float, stop_event=None):
    try:
        item = request_queue.get(timeout=wait_seconds)
    except queue.Empty:
        return []
    if item[0] == "stop":
        if stop_event is not None:
            stop_event.set()
        return None
    requests = [item]
    positions = len(item[3][0])
    deadline = time.monotonic() + wait_seconds
    while (stop_event is None or not stop_event.is_set()) and positions < gpu_batch_size:
        now = time.monotonic()
        if now >= deadline:
            break
        try:
            item = request_queue.get(timeout=max(0.0005, deadline - now))
        except queue.Empty:
            continue
        if item[0] == "stop":
            if stop_event is not None:
                stop_event.set()
            return None
        requests.append(item)
        positions += len(item[3][0])
    return requests

def _configure_cuda_process(device: str, torch_module, memory_fraction: float = 0.90) -> None:
    try:
        torch_module.set_num_threads(1)
        torch_module.set_num_interop_threads(1)
    except Exception:
        pass
    if device != "cuda":
        return
    torch_module.cuda.set_per_process_memory_fraction(memory_fraction)
    torch_module.backends.cuda.matmul.allow_tf32 = True
    torch_module.backends.cudnn.allow_tf32 = True
    torch_module.backends.cudnn.benchmark = True
    torch_module.set_float32_matmul_precision("high")

def _utilization_monitor_process_main(
    stop_event,
    games_completed,
    positions_evaluated,
    eval_batches,
    batch_positions,
    target_batch_size: int,
    log_path: str,
) -> None:
    _ensure_importable()
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    while not stop_event.is_set():
        time.sleep(30.0)
        gpu_util, vram_used, vram_total = _query_gpu_utilization()
        ram_used, ram_total, ram_available, swap_used, swap_total = _query_memory_utilization()
        
        # Read and print utilization parameters from the evaluator and monitor
        games = games_completed.value
        pos_eval = positions_evaluated.value
        batches = eval_batches.value
        avg_batch = batch_positions.value / max(1, batches)
        
        # Calculate rates over elapsed time
        elapsed_min = (time.monotonic() - start_time) / 60.0
        pos_per_min = pos_eval / max(0.01, elapsed_min)
        games_per_hour = (games / max(0.01, elapsed_min)) * 60.0
        
        line = (
            f"games={games} ({games_per_hour:.1f} g/hr) | "
            f"pos_eval={pos_eval} ({pos_per_min:.1f} p/min) | "
            f"batches={batches} avg_batch={avg_batch:.1f} | "
            f"gpu_util={gpu_util:.0f}% vram={vram_used}/{vram_total}MiB ram_avail={ram_available}MiB"
        )
        print(line, flush=True)

def _query_gpu_utilization() -> tuple[float, int, int]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5.0,
        )
        util, used, total = [part.strip() for part in output.splitlines()[0].split(",")]
        return float(util), int(used), int(total)
    except Exception:
        return 0.0, 0, 0

def _query_memory_utilization() -> tuple[int, int, int, int, int]:
    """High-performance memory query using psutil for cross-platform support (Windows/Linux)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        return (int(vm.used / 1024 / 1024),
                int(vm.total / 1024 / 1024),
                int(vm.available / 1024 / 1024),
                int(sw.used / 1024 / 1024),
                int(sw.total / 1024 / 1024))
    except ImportError:
        return 0, 131072, 120000, 0, 0

def _model_payload_for_process(model) -> tuple[dict, dict]:
    config = asdict(model.config) if hasattr(model, "config") else {}
    state_dict = {}
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu()
        try:
            value.share_memory_()
        except RuntimeError:
            pass
        state_dict[name] = value
    return config, state_dict

def _self_play_process_main(
    worker_id: int,
    request_queue,
    response_queue,
    game_queue,
    config: SelfPlayConfig,
    rng_seed: int | None,
    active_workers,
) -> None:
    _ensure_importable()
    active_workers[worker_id] = True
    try:
        rng = random.Random(rng_seed)
        evaluator = QueueEvaluatorProxy(worker_id, request_queue, response_queue)
        mcts = _make_mcts(evaluator, config)
        # Play exactly one game, put results in queue, reset, and exit cleanly [2]
        game_result = play_game(mcts, config, rng)
        game_queue.put((worker_id, game_result, None))
        mcts.reset()
        del mcts
    except Exception:
        game_queue.put((worker_id, None, traceback.format_exc()))
    finally:
        active_workers[worker_id] = False

def _gpu_evaluator_process_main(
    model_config: dict,
    state_dict: dict,
    device: str,
    request_queue,
    response_queues,
    gpu_batch_size: int,
    max_wait_ms: float,
    active_workers,
) -> None:
    _ensure_importable()
    from .model import ModelConfig, ZeroNet

    import torch
    _configure_cuda_process(device, torch, 0.95)

    model = ZeroNet(ModelConfig(**model_config)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    wait_seconds = max_wait_ms / 1000.0

    while True:
        requests = _collect_eval_requests_nonblocking(request_queue, gpu_batch_size, wait_seconds)
        if requests is None:
            return  # Clean stop signal received, terminate process
        if not requests:
            continue  # Idle timeout, loop back safely without evaluating

        flat_fens = []
        flat_legal_ucis = []
        for _, _, _, data in requests:
            fens, legal_ucis = data
            flat_fens.extend(fens)
            flat_legal_ucis.extend(legal_ucis)
        try:
            from .board import Board
            from .encoding import encode_boards, move_to_policy_index
            from .move import Move

            boards = [Board.from_fen(fen) for fen in flat_fens]
            tensors = encode_boards(boards, device="cpu")
            masks = torch.stack([_encode_move_mask_from_ucis(b, legal, device="cpu") for b, legal in zip(boards, flat_legal_ucis)])
            is_cuda = device == "cuda" and torch.cuda.is_available()
            if is_cuda:
                tensors = tensors.pin_memory().to(device, non_blocking=True)
                masks = masks.pin_memory().to(device, non_blocking=True)
            else:
                tensors = tensors.to(device)
                masks = masks.to(device)
            amp_dtype = torch.bfloat16 if is_cuda and torch.cuda.is_bf16_supported() else torch.float16
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=is_cuda):
                    out = model(tensors, masks, return_dict=True)
            raw_values = out["value"].squeeze(-1).detach().cpu().tolist()
            raw_uncertainties = out["uncertainty"].detach().cpu().tolist()
            raw_policies = out["policy"].detach().cpu()
            legal_probs = []
            for batch_idx, (board, legal) in enumerate(zip(boards, flat_legal_ucis)):
                indices = torch.tensor([move_to_policy_index(board, Move.from_uci(uci)) for uci in legal], dtype=torch.long)
                legal_probs.append(raw_policies[batch_idx][indices].tolist())
            error = None
        except Exception as exc:
            if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower() or "timeout" in str(exc).lower():
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[WARNING] GPU evaluator process forward pass failed: {exc}. Flushing CUDA cache and falling back to uniform evaluation.", flush=True)
                raw_values = [0.0] * len(flat_fens)
                raw_uncertainties = [1.0] * len(flat_fens)
                legal_probs = []
                for legal in flat_legal_ucis:
                    if not legal:
                        legal_probs.append([])
                    else:
                        prob = 1.0 / len(legal)
                        legal_probs.append([prob] * len(legal))
                error = None
            else:
                raw_values = raw_uncertainties = []
                legal_probs = []
                error = traceback.format_exc()

        offset = 0
        for _, worker_id, request_id, data in requests:
            batch_size = len(data[0])
            if error:
                response_queues[worker_id].put((request_id, (None,), error))
            else:
                response_queues[worker_id].put((
                    request_id,
                    (raw_values[offset:offset + batch_size],
                     raw_uncertainties[offset:offset + batch_size],
                     legal_probs[offset:offset + batch_size]),
                    None,
                ))
            offset += batch_size

def _adjudicate(values: list[float]) -> bool:
    if len(values) < 20:
        return False
    # Corrected: Adjudication respects asymmetric scale around the -1.0 draw baseline [2]
    avg = sum(values) / len(values)
    is_stable = (max(values) - min(values)) <= 0.05
    is_crushing = (avg > 0.70) or (avg < -2.70)
    return is_stable and is_crushing

# CUSTOM REWARD CALCULATIONS

def _aggression_score(board: Board) -> float:
    opponent = board.turn ^ 1
    squares = board.squares
    opponent_squares = []
    for sq in range(64):
        piece = squares[sq]
        if piece != "." and ((piece.isupper() and opponent == WHITE) or (piece.islower() and opponent == BLACK)):
            opponent_squares.append(sq)
            
    if not opponent_squares:
        return 0.0
    attacked = sum(1 for sq in opponent_squares if board.is_square_attacked(sq, board.turn))
    return attacked / len(opponent_squares)

def _momentum_reward(board: Board, move: Move) -> float:
    if move.is_capture:
        return MOMENTUM_REWARD
    return 0.0

def _panic_penalty(board: Board, move: Move) -> float:
    if move.is_capture:
        return PANIC_PENALTY
    return 0.0

# PERSISTENT TRAINING LOOP (MAIN)

def _trainer_process_main(
    model_config: dict,
    shared_state: dict,
    state_lock,
    weights_updated,
    stop_event,
    game_queue,
    generation_value,
    train_config_dict: dict,
    device: str,
    cuda_memory_fraction: float,
    updates_per_game: float,
    iteration: int,
    train_updates: int,
    elo: float,
    replay_path: str,
    cold_replay_path: str,
    checkpoint_dir: str,
    training_log_path: str,
    games_per_generation: int,
) -> None:
    _ensure_importable()
    import torch
    from .checkpoint import CheckpointManager
    from .ema import EMATeacher
    from .model import ModelConfig, ZeroNet
    from .training import ContinuousLRScheduler, TrainConfig, TrainingLogger, make_optimizer, train_step

    _configure_cuda_process(device, torch, cuda_memory_fraction)
    model = ZeroNet(ModelConfig(**model_config)).to(device)
    with state_lock:
        model.load_state_dict(shared_state)
    teacher = EMATeacher(model)
    train_config = TrainConfig(**train_config_dict)
    optimizer = make_optimizer(model, train_config)
    scheduler = ContinuousLRScheduler(optimizer, train_config.initial_lr, train_config.continuous_lr)
    logger = TrainingLogger(training_log_path)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    needs_scaler = device == "cuda" and not use_bf16
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    from .ewc import ElasticWeightConsolidation
    ewc = ElasticWeightConsolidation()
    replay_file = Path(replay_path)
    if replay_file.exists():
        replay = PrioritizedReplayBuffer.load(replay_file, cold_path=cold_replay_path)
    else:
        replay = PrioritizedReplayBuffer(cold_path=cold_replay_path)

    completed_games = 0
    # Sliding windows for rolling analytics
    plies_window = deque(maxlen=100)
    reasons_window = deque(maxlen=100)
    durations_window = deque(maxlen=100)

    try:
        while not stop_event.is_set():
            try:
                _, payload, error = game_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if error is not None:
                print(f"\n[ERROR] Worker process encountered an exception:\n{error}", flush=True)
                continue

            try:
                result, experiences, sans, reason, meta = payload
            except (ValueError, TypeError) as exc:
                print(f"[WARNING] Corrupted game payload from worker. Skipping. Error: {exc}", flush=True)
                continue
            replay.extend(experiences)
            completed_games += 1
            rated_side = WHITE if completed_games % 2 else BLACK
            elo, elo_delta = update_rating_from_result(elo, elo, result, rated_side)

            is_boundary = (completed_games % max(1, games_per_generation) == 0)
            if is_boundary:
                iteration += 1
                generation_value.value = iteration
            
            # Append to sliding windows
            plies_window.append(len(sans))
            reasons_window.append(reason)
            durations_window.append(meta["duration"])

            # Format a highly descriptive string for the game outcome
            if result == "1/2-1/2":
                if reason == "stalemate":
                    outcome_desc = "Draw by Stalemate"
                elif reason == "max_plies":
                    outcome_desc = "Draw by Max Plies"
                elif reason == "draw_material":
                    outcome_desc = "Draw by Dead Material"
                else:
                    outcome_desc = "Draw by Threefold Repetition"
            elif result == "1-0":
                if reason == "resignation":
                    outcome_desc = "White won by Resignation"
                elif reason == "adjudication":
                    outcome_desc = "White won by Adjudication"
                else:
                    outcome_desc = "White won by CHECKMATE"
            else: # "0-1"
                if reason == "resignation":
                    outcome_desc = "Black won by Resignation"
                elif reason == "adjudication":
                    outcome_desc = "Black won by Adjudication"
                else:
                    outcome_desc = "Black won by CHECKMATE"

            # Calculate Rolling 100-Game Percentages for the console log
            total_in_window = len(reasons_window)
            checkmate_pct = (reasons_window.count("checkmate") / total_in_window) * 100.0
            resign_pct = (reasons_window.count("resignation") / total_in_window) * 100.0
            draw_pct = (sum(1 for r in reasons_window if r in ("stalemate", "max_plies", "draw_material", "draw_repetition")) / total_in_window) * 100.0
            avg_plies = sum(plies_window) / total_in_window
            avg_seconds = sum(durations_window) / total_in_window

            # Print the highly descriptive, detailed game logging line
            print(
                f"[GAME #{completed_games:05d}] Plies: {len(sans):3d} ({avg_plies:.1f} avg) | "
                f"Reason: {outcome_desc:<25} | "
                f"Roll %: Mate={checkmate_pct:.1f}% Resign={resign_pct:.1f}% Draw={draw_pct:.1f}% | "
                f"Imbalance: {meta['material_delta']:+2d} | "
                f"Opening: {meta['opening']:<28} | "
                f"Duration: {meta['duration']:4.1f}s ({avg_seconds:.1f}s avg) | "
                f"Elo: {elo:.1f} ({elo_delta:+.1f}) | "
                f"Replay Buffer: {len(replay):,}",
                flush=True,
            )
            
            metrics = {}
            if len(replay) > 256:
                train_updates += 1
                metrics, scaler = train_step(
                    model,
                    optimizer,
                    replay,
                    train_config,
                    ewc=ewc,
                    iteration=train_updates,
                    scheduler=scheduler,
                    scaler=scaler,
                    logger=logger,
                )

                # Print active gradient update metrics directly alongside game completion logs
                print(
                    f"[TRAIN #{train_updates:05d}] Loss: {metrics.get('loss', 0.0):.4f} | "
                    f"Policy Loss: {metrics.get('policy_loss', 0.0):.4f} | "
                    f"Value Loss: {metrics.get('value_loss', 0.0):.4f} | "
                    f"Entropy: {metrics.get('policy_entropy', 0.0):.4f} | "
                    f"Value Error: {metrics.get('value_error', 0.0):.4f} | "
                    f"Aux Losses: Mat={metrics.get('material_loss', 0.0):.4f} Mob={metrics.get('mobility_loss', 0.0):.4f} King={metrics.get('king_safety_loss', 0.0):.4f} | "
                    f"LR: {metrics.get('lr', 0.0):.6g}",
                    flush=True,
                )
                teacher.update(model)
                _copy_state_to_shared(teacher.teacher, shared_state, state_lock)
                weights_updated.set()

            if completed_games % 100 == 0:
                # Drain pending games from the queue before arena to prevent worker backpressure
                while not game_queue.empty():
                    try:
                        _, pending_payload, pending_error = game_queue.get_nowait()
                    except queue.Empty:
                        break
                    if pending_error is not None:
                        print(f"\n[ERROR] Worker process encountered an exception:\n{pending_error}", flush=True)
                        continue
                    p_result, p_experiences, p_sans, p_reason, p_meta = pending_payload
                    replay.extend(p_experiences)
                    completed_games += 1
                    p_rated_side = WHITE if completed_games % 2 else BLACK
                    elo, p_elo_delta = update_rating_from_result(elo, elo, p_result, p_rated_side)
                    plies_window.append(len(p_sans))
                    reasons_window.append(p_reason)
                    durations_window.append(p_meta["duration"])
                    _append_training_game_history(
                        game_number=completed_games,
                        generation=iteration,
                        result=p_result,
                        moves_san=p_sans,
                        elo_after=elo,
                        elo_delta=p_elo_delta,
                        rated_side=p_rated_side,
                        replay_size=len(replay),
                        train_step=train_updates,
                        metrics={},
                    )

                print("\n" + " "*20 + " [ARENA EVALUATION] " + " "*20, flush=True)
                print(f"Starting Arena Match: Student vs. EMA Teacher (Iteration {iteration})", flush=True)
                print("Games: 20 | MCTS Simulations: 160 | Device: " + str(device), flush=True)

                from .arena import play_arena
                from .mcts import NetworkEvaluator

                student_eval = NetworkEvaluator(model, device)
                teacher_eval = NetworkEvaluator(teacher.teacher, device)

                arena = play_arena(
                    student_eval,
                    teacher_eval,
                    games=20,
                    simulations=160,
                    iteration=iteration,
                    elo_a=elo,
                    log_path="logs/arena.log",
                )

                elo = float(arena["elo_a"])
                print(f"Results: Student Score: {arena['student_score']} | Teacher Score: {arena['teacher_score']}", flush=True)
                print(f"Match Stats: Student Wins: {arena['wins_a']} | Teacher Wins: {arena['wins_b']} | Draws: {arena['draws']}", flush=True)

                if arena["promote"]:
                    teacher.promote(model)
                    print(f"Status: [PROMOTED] Student defeated Teacher (>60% score)! Student is now the new EMA Teacher.", flush=True)
                else:
                    print(f"Status: [RETAINED] Teacher retained its title (Student score <=60%).", flush=True)
                print(f"New Running Elo: {elo:.1f}", flush=True)
                print("\n", flush=True)

            if completed_games % 50 == 0 and len(replay) > 0:
                ewc.consolidate(model, replay, device=device)

            if is_boundary:
                replay.save(replay_path)
                checkpoint_manager.save(model, iteration, elo, optimizer.state_dict())

            # Append completed self-play game history to PGN and JSONL files for the UI
            _append_training_game_history(
                game_number=completed_games,
                generation=iteration,
                result=result,
                moves_san=sans,
                elo_after=elo,
                elo_delta=elo_delta,
                rated_side=rated_side,
                replay_size=len(replay),
                train_step=train_updates,
                metrics=metrics,
            )

            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        try:
            replay.save(replay_path)
            checkpoint_manager.save(model, iteration, elo, optimizer.state_dict())
            print(f"[TRAIN] Saved final checkpoint and replay buffer at iteration {iteration}.", flush=True)
        except Exception as e:
            print(f"[TRAIN] Error saving final checkpoint: {e}", flush=True)

def _copy_state_to_shared(model, shared_state: dict, state_lock) -> None:
    with state_lock:
        for name, tensor in model.state_dict().items():
            shared_state[name].copy_(tensor.detach().cpu())

def _append_training_game_history(
    game_number: int,
    generation: int,
    result: str,
    moves_san: list[str],
    elo_after: float,
    elo_delta: float,
    rated_side: int,
    replay_size: int,
    train_step: int,
    metrics: dict[str, float],
    jsonl_path: str | Path = "data/training_games.jsonl",
    pgn_path: str | Path = "data/training_games.pgn",
) -> None:
    timestamp = datetime.now(timezone.utc)
    game_id = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{game_number:07d}"
    pgn = export_pgn(
        moves_san,
        result,
        {
            "Event": "ZERO Training Self-Play",
            "Site": "ZERO Local",
            "Date": timestamp.strftime("%Y.%m.%d"),
            "Round": str(generation),
            "White": "ZERO",
            "Black": "ZERO",
        },
    )
    record = {
        "id": game_id,
        "timestamp": timestamp.isoformat(),
        "game_number": game_number,
        "generation": generation,
        "result": result,
        "elo_after": float(elo_after),
        "elo_delta": float(elo_delta),
        "rated_side": "white" if rated_side == WHITE else "black",
        "replay_size": int(replay_size),
        "train_step": int(train_step),
        "ply_count": len(moves_san),
        "moves_san": moves_san,
        "loss": float(metrics.get("loss", 0.0)),
        "pgn": pgn,
    }
    jsonl_file = Path(jsonl_path)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    pgn_file = Path(pgn_path)
    pgn_file.parent.mkdir(parents=True, exist_ok=True)
    with pgn_file.open("a", encoding="utf-8") as fh:
        fh.write(pgn + "\n\n")

def main(argv: list[str] | None = None) -> None:
    """CLI entry point: generate self-play games and save PGN output."""
    parser = argparse.ArgumentParser(description="Generate self-play games using a ZERO checkpoint.")
    parser.add_argument("--checkpoint", help="Path to model checkpoint. If not specified, UniformEvaluator is used.")
    parser.add_argument("--games", type=int, default=2, help="Number of games to generate.")
    parser.add_argument("--simulations", type=int, default=200, help="Number of MCTS simulations per move.")
    parser.add_argument("--batch-size", type=int, default=24, help="MCTS batch size.")
    parser.add_argument("--max-plies", type=int, default=512, help="Maximum plies per game.")
    parser.add_argument("--device", default="cpu", help="Device to run on ('cpu' or 'cuda').")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--out-pgn", default="data/selfplay.pgn", help="File to write PGN output to.")
    parser.add_argument("--gpu-batch-size", type=int, default=64, help="GPU evaluator batch size for multiprocess mode.")
    parser.add_argument("--max-wait-ms", type=float, default=20.0, help="Max wait time in ms for GPU batch coalescing.")
    args = parser.parse_args(argv)

    if args.checkpoint:
        from .model import load_model
        model = load_model(args.checkpoint, args.device)
    else:
        model = None

    config = SelfPlayConfig(
        simulations=args.simulations,
        batch_size=args.batch_size,
        max_plies=args.max_plies,
    )

    print(f"Generating {args.games} self-play games on {args.device}...")
    
    if args.device == "cuda" and model is not None:
        games_data = generate_multiprocess_games(
            model,
            device=args.device,
            games=args.games,
            config=config,
            rng_seed=args.seed,
            gpu_batch_size=args.gpu_batch_size,
            max_wait_ms=args.max_wait_ms,
        )
    else:
        if model is not None:
            evaluator = NetworkEvaluator(model, args.device)
        else:
            evaluator = UniformEvaluator()
        games_data = generate_parallel_games(
            evaluator,
            games=args.games,
            config=config,
            rng_seed=args.seed,
        )

    from .pgn import export_pgn
    from pathlib import Path
    pgn_path = Path(args.out_pgn)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pgn_path.open("a", encoding="utf-8") as fh:
        for idx, (result, experiences, sans, reason, meta) in enumerate(games_data):
            pgn_str = export_pgn(
                sans,
                result,
                {
                    "Event": "ZERO CLI Self-Play",
                    "Round": str(idx + 1),
                    "White": "ZERO",
                    "Black": "ZERO",
                }
            )
            fh.write(pgn_str + "\n\n")
            print(f"Game {idx + 1}: Result={result}, Reason={reason}, Plies={len(sans)}, Duration={meta['duration']:.1f}s")
            
    print(f"Saved PGN to {args.out_pgn}")