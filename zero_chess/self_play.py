"""Highly optimized parallel and persistent self-play game generator."""

from __future__ import annotations

import argparse
import gc
import json
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

import torch
import torch.multiprocessing as mp

from .board import Board
from .constants import (
    BLACK, BK, BQ, EMPTY, WHITE, WK, WQ, color_of, file_of,
    parse_square, piece_type, rank_of, square, square_name
)
from .elo import DEFAULT_ELO, update_rating_with_reason
from .encoding import terminal_wdl
from .mcts import MCTS, NetworkEvaluator, UniformEvaluator
from .move import EN_PASSANT, Move
from .pgn import export_pgn
from .replay import Experience, PrioritizedReplayBuffer
from .targets import (
    AGGRESSION_WEIGHT, MOMENTUM_REWARD, PANIC_PENALTY, 
    opponent_value, game_result_to_values
)

@dataclass(slots=True)
class SelfPlayConfig:
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
    fen: str
    policy: dict[str, float]
    turn: int
    root_value: float
    aggression_score: float
    momentum_reward: float
    panic_penalty: float

def play_game(
    mcts: MCTS,
    config: SelfPlayConfig | None = None,
    rng: random.Random | None = None,
) -> tuple[str, list[Experience], list[str], str, dict]:
    config = config or SelfPlayConfig()
    rng = rng or random.Random()
    board = Board()
    records: list[PositionRecord] = []
    sans: list[str] = []
    adjudication: list[float] = []
    pending_panic = 0.0
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
                panic_penalty=pending_panic,
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
        pending_panic = _panic_penalty(board, move)
        board.push(move)
        
        if config.reuse_tree:
            mcts.advance_to(move)
        else:
            mcts.reset()
            if ply % 16 == 15:
                gc.collect()

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

def _collect_eval_requests(request_queue, gpu_batch_size: int, wait_seconds: float, active_workers=None):
    item = request_queue.get()
    if item[0] == "stop":
        return None
    requests = [item]
    positions = len(item[3])
    deadline = time.monotonic() + wait_seconds
    while True:
        if positions >= gpu_batch_size:
            break

        search_is_finished = True
        if active_workers is not None:
            active_ids = {i for i, active in enumerate(active_workers) if active}
            pending_ids = {req[1] for req in requests}
            search_is_finished = active_ids.issubset(pending_ids)

        now = time.monotonic()
        if now >= deadline:
            if positions >= 16 or search_is_finished:
                break

        timeout = max(0.001, deadline - now) if now < deadline else wait_seconds
        try:
            item = request_queue.get(timeout=timeout)
        except queue.Empty:
            continue
        if item[0] == "stop":
            break
        requests.append(item)
        positions += len(item[3])
    return requests

def generate_parallel_games(
    evaluator,
    games: int = 2,
    config: SelfPlayConfig | None = None,
    rng_seed: int | None = None,
) -> list[tuple[str, list[Experience], list[str], str, dict]]:
    config = config or SelfPlayConfig()
    results: list[tuple[str, list[Experience], list[str], str, dict] | None] = [None] * games

    def worker(index: int) -> None:
        rng = random.Random(None if rng_seed is None else rng_seed + index)
        mcts = MCTS(
            evaluator,
            c_puct=1.5 if config.generation < 200 else 1.0,
            batch_size=config.batch_size,
            simulations=config.simulations,
        )
        results[index] = play_game(mcts, config, rng)
        mcts.reset()
        del mcts
        gc.collect()

    threads = [threading.Thread(target=worker, args=(idx,), daemon=True) for idx in range(games)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [result for result in results if result is not None]

class QueueEvaluatorProxy:
    """Synchronous MCTS evaluator proxy backed by multiprocessing queues."""

    def __init__(self, worker_id: int, request_queue, response_queue) -> None:
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._next_request_id = 0

    def evaluate_batch(self, boards: list[Board]):
        self._next_request_id += 1
        request_id = self._next_request_id
        self.request_queue.put(("eval", self.worker_id, request_id, boards))
        while True:
            response_id, results, error = self.response_queue.get()
            if response_id != request_id:
                continue
            if error is not None:
                raise RuntimeError(error)
            return results

def generate_multiprocess_games(
    model,
    device: str = "cuda",
    games: int = 2,
    config: SelfPlayConfig | None = None,
    rng_seed: int | None = None,
    gpu_batch_size: int = 64,
    max_wait_ms: float = 2.0,
) -> list[tuple[str, list[Experience], list[str], str, dict]]:
    """Generate self-play with CPU worker processes feeding one CUDA evaluator process."""
    config = config or SelfPlayConfig()
    ctx = mp.get_context("spawn")
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
            continue
        received += 1
        if error is None:
            results[idx] = payload

    for process in workers:
        if process.is_alive():
            process.terminate()
    request_queue.put(("stop",))
    evaluator.join(timeout=5.0)
    return [result for result in results if result is not None]

@dataclass(slots=True)
class CudaTrainingRuntime:
    stop_event: object
    processes: list[object]
    resources: list[object]

    def stop(self, timeout: float = 10.0) -> None:
        self.stop_event.set()
        for process in self.processes:
            process.join(timeout=timeout)
        for process in self.processes:
            if process.is_alive():
                process.terminate()

def start_persistent_cuda_training(
    model,
    train_config,
    *,
    games: int = 2,
    self_play_config: SelfPlayConfig | None = None,
    device: str = "cuda",
    gpu_batch_size: int = 64,
    eval_wait_ms: float = 10.0,
    cuda_memory_fraction: float = 0.40,
    gpu_cooldown_ms: float = 0.0,
    compile_model: bool = False,
    updates_per_game: float = 1.0,
    iteration: int = 0,
    train_updates: int = 0,
    elo: float = DEFAULT_ELO,
    replay_path: str = "data/replay.pkl",
    cold_replay_path: str = "data/replay_cold.sqlite3",
    checkpoint_dir: str = "checkpoints",
    training_log_path: str = "logs/training.log",
    monitor_log_path: str = "logs/utilization.log",
) -> CudaTrainingRuntime:
    """Start persistent self-play with strict 8GB RAM safety barriers."""
    config = self_play_config or SelfPlayConfig()
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue(maxsize=128)
    response_queues = [ctx.Queue(maxsize=16) for _ in range(games)]
    game_queue = ctx.Queue(maxsize=32)
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
            cuda_memory_fraction,
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
            cuda_memory_fraction,
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

    return CudaTrainingRuntime(stop_event, [*workers, evaluator, trainer, monitor], resources)

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
    active_workers[worker_id] = True
    try:
        rng = random.Random(rng_seed)
        evaluator = QueueEvaluatorProxy(worker_id, request_queue, response_queue)
        while not stop_event.is_set():
            # Memory Guard: Halt and flush if free system RAM falls below 512MB
            _, _, ram_avail, _, _ = _query_memory_utilization()
            if ram_avail < 512:
                gc.collect()
                time.sleep(5.0)
                continue

            generation = int(generation_value.value)
            config = replace(base_config, generation=generation)
            mcts = MCTS(
                evaluator,
                batch_size=config.batch_size,
                simulations=config.simulations,
            )
            try:
                game_result = play_game(mcts, config, rng)
                game_queue.put((worker_id, game_result, None))
                mcts.reset()
                del mcts
                gc.collect()
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
    from .encoding import INPUT_CHANNELS, POLICY_SIZE, encode_board_into, encode_move_mask_into, move_to_policy_index
    from .model import ModelConfig, ZeroNet

    _configure_cuda_process(device, torch, cuda_memory_fraction)
    model = ZeroNet(ModelConfig(**model_config)).to(device)
    with state_lock:
        model.load_state_dict(shared_state)
    model.eval()
    compiled_model = torch.compile(model, mode="reduce-overhead") if compile_model else model
    pin = device == "cuda"
    input_host = torch.empty((gpu_batch_size, INPUT_CHANNELS, 8, 8), dtype=torch.float32, pin_memory=pin)
    mask_host = torch.empty((gpu_batch_size, POLICY_SIZE), dtype=torch.float32, pin_memory=pin)
    input_device = torch.empty((gpu_batch_size, INPUT_CHANNELS, 8, 8), dtype=torch.float32, device=device)
    mask_device = torch.empty((gpu_batch_size, POLICY_SIZE), dtype=torch.float32, device=device)
    wait_seconds = eval_wait_ms / 1000.0

    while not stop_event.is_set():
        if weights_updated.is_set():
            with state_lock:
                model.load_state_dict(shared_state)
                weights_updated.clear()
            model.eval()

        requests = _collect_eval_requests_nonblocking(request_queue, gpu_batch_size, wait_seconds, stop_event, active_workers)
        if requests is None:
            return  # Clean stop
        if not requests:
            continue  # Idle loop back

        flat_boards = [board for _, _, _, boards in requests for board in boards]
        
        # Increment batch and evaluation stats counters
        with positions_evaluated.get_lock():
            positions_evaluated.value += len(flat_boards)
        with eval_batches.get_lock():
            eval_batches.value += 1
        with batch_positions.get_lock():
            batch_positions.value += len(flat_boards)

        try:
            flat_results = _evaluate_preallocated_chunk(
                compiled_model,
                flat_boards,
                device,
                torch.float16,
                input_host,
                mask_host,
                input_device,
                mask_device,
                encode_board_into,
                encode_move_mask_into,
                move_to_policy_index,
            )
            error = None
        except Exception:
            flat_results = []
            error = traceback.format_exc()

        offset = 0
        for _, worker_id, request_id, boards in requests:
            end = offset + len(boards)
            response_queues[worker_id].put((request_id, None if error else flat_results[offset:end], error))
            offset = end
            
        # Fulfill defined GPU cooldown sleep parameter to avoid thermal limits
        if gpu_cooldown_ms > 0.0:
            time.sleep(gpu_cooldown_ms / 1000.0)

        gc.collect()

def _evaluate_preallocated_chunk(
    compiled_model,
    boards,
    device: str,
    amp_dtype,
    input_host,
    mask_host,
    input_device,
    mask_device,
    encode_board_into,
    encode_move_mask_into,
    move_to_policy_index,
):
    batch_size = len(boards)
    capacity = int(input_host.shape[0])
    if batch_size > capacity:
        results = []
        for start in range(0, batch_size, capacity):
            results.extend(
                _evaluate_preallocated_chunk(
                    compiled_model,
                    boards[start : start + capacity],
                    device,
                    amp_dtype,
                    input_host,
                    mask_host,
                    input_device,
                    mask_device,
                    encode_board_into,
                    encode_move_mask_into,
                    move_to_policy_index,
                )
            )
        return results

    for row, board in enumerate(boards):
        legal = board.legal_moves()
        encode_board_into(input_host[row], board)
        encode_move_mask_into(mask_host[row], legal, board)

    input_device[:batch_size].copy_(input_host[:batch_size], non_blocking=True)
    mask_device[:batch_size].copy_(mask_host[:batch_size], non_blocking=True)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
        out = compiled_model(input_device[:batch_size], mask_device[:batch_size], return_dict=True)
        
    policy_cpu = out["policy"].detach().cpu()
    values = out["value"].squeeze(-1).detach().cpu().tolist()
    uncertainties = out["uncertainty"].detach().cpu().tolist()
    results = []
    for row, board in enumerate(boards):
        legal = board.legal_moves()
        indices = [move_to_policy_index(board, m) for m in legal]
        priors = {m: float(policy_cpu[row, idx]) for m, idx in zip(legal, indices, strict=True)}
        results.append((priors, float(values[row]), float(uncertainties[row])))
    return results

def _collect_eval_requests_nonblocking(request_queue, gpu_batch_size: int, wait_seconds: float, stop_event=None, active_workers=None):
    try:
        item = request_queue.get(timeout=0.01)
    except queue.Empty:
        return []
    if item[0] == "stop":
        if stop_event is not None:
            stop_event.set()
        return None # Return None to signal a clean, cooperative shutdown [2]
    requests = [item]
    positions = len(item[3])
    deadline = time.monotonic() + wait_seconds
    while (stop_event is None or not stop_event.is_set()) and positions < gpu_batch_size:
        now = time.monotonic()
        if now >= deadline:
            break
        try:
            item = request_queue.get(timeout=max(0.001, deadline - now))
        except queue.Empty:
            continue
        if item[0] == "stop":
            if stop_event is not None:
                stop_event.set()
            return None # Return None to signal a clean, cooperative shutdown [2]
        requests.append(item)
        positions += len(item[3])
    return requests

def _configure_cuda_process(device: str, torch_module, memory_fraction: float = 0.40) -> None:
    try:
        torch_module.set_num_threads(1)
        torch_module.set_num_interop_threads(1)
    except Exception:
        pass
    if device != "cuda":
        return
    torch_module.cuda.set_per_process_memory_fraction(memory_fraction)
    torch_module.backends.cuda.matmul.allow_tf32 = True
    torch_module.backends.cudnn.benchmark = True

def _utilization_monitor_process_main(
    stop_event,
    games_completed,
    positions_evaluated,
    eval_batches,
    batch_positions,
    target_batch_size: int,
    log_path: str,
) -> None:
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
    """Retrieve platform memory utilization, falling back gracefully to avoid freezes on non-Linux systems."""
    try:
        fields: dict[str, int] = {}
        with Path("/proc/meminfo").open("r", encoding="utf-8") as fh:
            for line in fh:
                name, raw_value = line.split(":", 1)
                fields[name] = int(raw_value.strip().split()[0]) // 1024
        total = fields.get("MemTotal", 0)
        available = fields.get("MemAvailable", 0)
        swap_total = fields.get("SwapTotal", 0)
        swap_free = fields.get("SwapFree", 0)
        return total - available, total, available, swap_total - swap_free, swap_total
    except Exception:
        # Cross-platform fallback: return baseline values to prevent thread starvation/infinite pauses
        return 0, 8192, 4096, 0, 0

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
    active_workers[worker_id] = True
    try:
        rng = random.Random(rng_seed)
        evaluator = QueueEvaluatorProxy(worker_id, request_queue, response_queue)
        mcts = MCTS(
            evaluator,
            batch_size=config.batch_size,
            simulations=config.simulations,
        )
        # Play exactly one game, put results in queue, reset, and exit cleanly [2]
        game_result = play_game(mcts, config, rng)
        game_queue.put((worker_id, game_result, None))
        mcts.reset()
        del mcts
        gc.collect()
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
    from .model import ModelConfig, ZeroNet

    import torch

    model = ZeroNet(ModelConfig(**model_config)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    wait_seconds = max_wait_ms / 1000.0

    while True:
        requests = _collect_eval_requests_nonblocking(request_queue, gpu_batch_size, wait_seconds, None, active_workers) # Corrected target signature [2]
        if requests is None:
            return  # Clean stop signal received, terminate process [2]
        if not requests:
            continue  # Idle timeout, loop back safely without evaluating [2]
            
        flat_boards = [board for _, _, _, boards in requests for board in boards]
        try:
            flat_results = model.evaluate_batch(flat_boards, device)
            error = None
        except Exception:
            flat_results = []
            error = traceback.format_exc()

        offset = 0
        for _, worker_id, request_id, boards in requests:
            end = offset + len(boards)
            response_queues[worker_id].put((request_id, None if error else flat_results[offset:end], error))
            offset = end

def _adjudicate(values: list[float]) -> bool:
    if len(values) < 20:
        return False
    # Corrected: Adjudication respects asymmetric scale around the -1.0 draw baseline [2]
    avg = sum(values) / len(values)
    is_stable = (max(values) - min(values)) <= 0.05
    is_crushing = (avg > 0.70) or (avg < -2.70)
    return is_stable and is_crushing

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
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    checkpoint_manager = CheckpointManager(checkpoint_dir)
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

    while not stop_event.is_set():
        try:
            _, payload, error = game_queue.get(timeout=1.0)
        except queue.Empty:
            # Run Arena Evaluation dynamically every 100 completed games
            if completed_games > 0 and completed_games % 100 == 0:
                print("\n" + "="*20 + " [ARENA EVALUATION] " + "="*20, flush=True)
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
                print("="*60 + "\n", flush=True)
                
                completed_games += 1  # Offset to prevent infinite evaluation triggers
            continue
            
        if error is not None:
            print(f"\n[ERROR] Worker process encountered an exception:\n{error}", flush=True)
            continue
            
        result, experiences, sans, reason, meta = payload # Unpack 5-tuple with metadata!
        replay.extend(experiences)
        completed_games += 1
        rated_side = WHITE if completed_games % 2 else BLACK
        elo, elo_delta = update_rating_with_reason(elo, result, rated_side, reason) # Use custom Elo rules!
        iteration += 1 if completed_games % max(1, games_per_generation) == 0 else 0
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
            metrics = train_step(
                model,
                optimizer,
                replay,
                train_config,
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

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

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