"""Self-play game generation and online experience creation."""

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
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import torch.multiprocessing as mp

from .board import Board
from .constants import BLACK, BK, BQ, EMPTY, WHITE, WK, WQ, color_of, file_of, parse_square, piece_type, rank_of, square, square_name
from .elo import DEFAULT_ELO, update_rating_from_result
from .encoding import terminal_wdl
from .mcts import MCTS, NetworkEvaluator, UniformEvaluator
from .move import EN_PASSANT, Move
from .pgn import export_pgn
from .replay import Experience, PrioritizedReplayBuffer
from .targets import AGGRESSION_WEIGHT, MOMENTUM_REWARD, PANIC_PENALTY, game_result_to_value, opponent_value


@dataclass(slots=True)
class SelfPlayConfig:
    simulations: int = 200
    batch_size: int = 24
    max_plies: int = 1024
    temperature_moves: int = 30
    opening_random_plies: int = 6
    opening_random_prob: float = 0.5
    resign_value: float = -0.95
    disable_resign: bool = False
    generation: int = 0
    symmetry_augmentation: bool = True
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
) -> tuple[str, list[Experience], list[str]]:
    config = config or SelfPlayConfig()
    rng = rng or random.Random()
    board = Board()
    records: list[PositionRecord] = []
    sans: list[str] = []
    adjudication: list[float] = []
    pending_panic = 0.0

    for ply in range(config.max_plies):
        result = board.outcome()
        if result is not None:
            return result, _finalize(records, result, rng, config.symmetry_augmentation), sans

        temperature = 1.0 if ply < config.temperature_moves else 0.0
        search = mcts.search(
            board,
            num_simulations=config.simulations,
            temperature=temperature,
            add_noise=True,
            generation=config.generation,
        )
        if search.resigned and not config.disable_resign:
            result = "0-1" if board.turn == 0 else "1-0"
            return result, _finalize(records, result, rng, config.symmetry_augmentation), sans

        legal = board.legal_moves()
        if ply < config.opening_random_plies and rng.random() < config.opening_random_prob and legal:
            move = rng.choice(legal)
        else:
            move = search.move or rng.choice(legal)

        policy = search.policy
        root_value = search.root.q
        momentum = _momentum_reward(board, move)
        records.append(
            PositionRecord(
                fen=board.fen(),
                policy={move_.uci(): prob for move_, prob in policy.items()},
                turn=board.turn,
                root_value=root_value,
                aggression_score=_aggression_score(board),
                momentum_reward=momentum,
                panic_penalty=pending_panic,
            )
        )

        adjudication.append(root_value)
        if len(adjudication) >= 20 and _adjudicate(adjudication[-20:]):
            winning_side_is_turn = sum(adjudication[-20:]) > 0
            if winning_side_is_turn:
                result = "1-0" if board.turn == 0 else "0-1"
            else:
                result = "0-1" if board.turn == 0 else "1-0"
            return result, _finalize(records, result, rng, config.symmetry_augmentation), sans

        sans.append(board.san(move))
        pending_panic = _panic_penalty(board, move)
        board.push(move)
        if config.reuse_tree:
            mcts.advance_to(move)
        else:
            mcts.reset()
            if ply % 16 == 15:
                gc.collect()

    return "1/2-1/2", _finalize(records, "1/2-1/2", rng, config.symmetry_augmentation), sans


def generate_parallel_games(
    evaluator,
    games: int = 4,
    config: SelfPlayConfig | None = None,
    rng_seed: int | None = None,
) -> list[tuple[str, list[Experience], list[str]]]:
    config = config or SelfPlayConfig()
    results: list[tuple[str, list[Experience], list[str]] | None] = [None] * games

    def worker(index: int) -> None:
        rng = random.Random(None if rng_seed is None else rng_seed + index)
        mcts = MCTS(
            evaluator,
            c_puct=1.5 if config.generation < 200 else 1.0,
            batch_size=config.batch_size,
            add_noise=True,
            resign_threshold=-1.0 if config.generation < 10 else config.resign_value,
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
    games: int = 6,
    config: SelfPlayConfig | None = None,
    rng_seed: int | None = None,
    gpu_batch_size: int = 256,
    max_wait_ms: float = 2.0,
) -> list[tuple[str, list[Experience], list[str]]]:
    """Generate self-play with CPU worker processes feeding one CUDA evaluator process."""

    config = config or SelfPlayConfig()
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue(maxsize=max(32, games * 8))
    response_queues = [ctx.Queue(maxsize=8) for _ in range(games)]
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

    results: list[tuple[str, list[Experience], list[str]] | None] = [None] * games
    errors: list[str] = []
    received = 0
    while received < games:
        try:
            idx, payload, error = result_queue.get(timeout=1.0)
        except queue.Empty:
            failed = [process for process in workers if process.exitcode not in (None, 0)]
            if failed:
                errors.append(f"self-play worker exited with code {failed[0].exitcode}")
                break
            if evaluator.exitcode not in (None, 0):
                errors.append(f"gpu evaluator exited with code {evaluator.exitcode}")
                break
            continue
        received += 1
        if error is not None:
            errors.append(error)
        else:
            results[idx] = payload

    for process in workers:
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
    request_queue.put(("stop",))
    evaluator.join(timeout=10.0)
    if evaluator.is_alive():
        evaluator.terminate()
        evaluator.join(timeout=2.0)

    if errors:
        raise RuntimeError("\n".join(errors))
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
                process.join(timeout=2.0)


def start_persistent_cuda_training(
    model,
    train_config,
    *,
    games: int = 6,
    self_play_config: SelfPlayConfig | None = None,
    device: str = "cuda",
    gpu_batch_size: int = 256,
    eval_wait_ms: float = 50.0,
    cuda_memory_fraction: float = 0.70,
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
    """Start persistent CPU self-play workers, one CUDA evaluator, one trainer, and one monitor."""

    config = self_play_config or SelfPlayConfig()
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue(maxsize=max(128, games * 16))
    response_queues = [ctx.Queue(maxsize=16) for _ in range(games)]
    game_queue = ctx.Queue(maxsize=max(32, games * 4))
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
    resources = [
        request_queue,
        *response_queues,
        game_queue,
        weights_updated,
        state_lock,
        generation_value,
        games_completed,
        positions_evaluated,
        eval_batches,
        batch_positions,
        shared_state,
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
            # Memory Guard: if ram_avail < 1GB, pause game generation, call gc.collect()
            _, _, ram_avail, _, _ = _query_memory_utilization()
            if ram_avail < 1024:
                print(f"[Worker {worker_id}] WARNING: Low memory detected ({ram_avail}MB available). Pausing game generation and clearing caches...", flush=True)
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                while not stop_event.is_set():
                    _, _, ram_avail, _, _ = _query_memory_utilization()
                    if ram_avail >= 1024:
                        break
                    time.sleep(5.0)
                if stop_event.is_set():
                    break
                print(f"[Worker {worker_id}] Memory recovered ({ram_avail}MB available). Resuming game generation.", flush=True)

            generation = int(generation_value.value)
            config = replace(base_config, generation=generation)
            mcts = MCTS(
                evaluator,
                c_puct=1.5 if generation < 200 else 1.0,
                batch_size=config.batch_size,
                add_noise=True,
                resign_threshold=-1.0 if generation < 10 else config.resign_value,
            )
            try:
                game_result = play_game(mcts, config, rng)
                game_queue.put((worker_id, game_result, None))
                mcts.reset()
                del mcts
                del game_result
                gc.collect()
                with games_completed.get_lock():
                    games_completed.value += 1
            except BaseException:
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

    import torch

    _configure_cuda_process(device, torch, cuda_memory_fraction)
    model = ZeroNet(ModelConfig(**model_config)).to(device)
    with state_lock:
        model.load_state_dict(shared_state)
    model.eval()
    compiled_model = torch.compile(model, mode="reduce-overhead") if compile_model and hasattr(torch, "compile") else model
    pin = device == "cuda"
    input_host = torch.empty((gpu_batch_size, INPUT_CHANNELS, 8, 8), dtype=torch.float32, pin_memory=pin)
    mask_host = torch.empty((gpu_batch_size, POLICY_SIZE), dtype=torch.float32, pin_memory=pin)
    input_device = torch.empty((gpu_batch_size, INPUT_CHANNELS, 8, 8), dtype=torch.float32, device=device)
    mask_device = torch.empty((gpu_batch_size, POLICY_SIZE), dtype=torch.float32, device=device)
    wait_seconds = eval_wait_ms / 1000.0
    cooldown_seconds = max(0.0, gpu_cooldown_ms) / 1000.0
    amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    while not stop_event.is_set():
        if weights_updated.is_set():
            with state_lock:
                model.load_state_dict(shared_state)
                weights_updated.clear()
            model.eval()

        requests = _collect_eval_requests_nonblocking(request_queue, gpu_batch_size, wait_seconds, stop_event, active_workers)
        if not requests:
            continue
        flat_boards = [board for _, _, _, boards in requests for board in boards]
        flat_results = []
        error = None
        try:
            buffer_capacity = int(input_host.shape[0])
            for start in range(0, len(flat_boards), buffer_capacity):
                chunk = flat_boards[start : start + buffer_capacity]
                flat_results.extend(
                    _evaluate_preallocated_chunk(
                        compiled_model,
                        chunk,
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
                with positions_evaluated.get_lock():
                    positions_evaluated.value += len(chunk)
                with eval_batches.get_lock():
                    eval_batches.value += 1
                with batch_positions.get_lock():
                    batch_positions.value += len(chunk)
                if cooldown_seconds:
                    time.sleep(cooldown_seconds)
        except BaseException:
            flat_results = []
            error = traceback.format_exc()

        offset = 0
        for _, worker_id, request_id, boards in requests:
            end = offset + len(boards)
            response_queues[worker_id].put((request_id, None if error else flat_results[offset:end], error))
            offset = end

        # Explicitly clear variables from the current batch to prevent leaks
        try:
            del flat_boards
            del flat_results
            del requests
        except NameError:
            pass
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
    import torch

    legal_moves = []
    policy_indices = []
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
        legal_moves.append(legal)
        policy_indices.append([move_to_policy_index(board, move) for move in legal])
        encode_board_into(input_host[row], board)
        encode_move_mask_into(mask_host[row], legal, board)

    input_device[:batch_size].copy_(input_host[:batch_size], non_blocking=True)
    mask_device[:batch_size].copy_(mask_host[:batch_size], non_blocking=True)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
        out = compiled_model(input_device, mask_device, return_dict=True)
    policy_cpu = out["policy"][:batch_size].detach().cpu()
    values = out["value"][:batch_size].squeeze(-1).detach().cpu().tolist()
    uncertainties = out["uncertainty"][:batch_size].detach().cpu().tolist()
    results = []
    for row, (legal, indices, value, uncertainty) in enumerate(zip(legal_moves, policy_indices, values, uncertainties, strict=True)):
        priors = {move: float(policy_cpu[row, index]) for move, index in zip(legal, indices, strict=True)}
        results.append((priors, float(value), float(uncertainty)))
    return results


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

    import torch

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
    last_checkpoint = time.monotonic()
    try:
        while not stop_event.is_set():
            # Memory Guard: if ram_avail < 1GB, pause and perform GC
            _, _, ram_avail, _, _ = _query_memory_utilization()
            if ram_avail < 1024:
                print(f"[Trainer] WARNING: Low memory detected ({ram_avail}MB available). Performing GC and waiting...", flush=True)
                gc.collect()
                if device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                while not stop_event.is_set():
                    _, _, ram_avail, _, _ = _query_memory_utilization()
                    if ram_avail >= 1024:
                        break
                    time.sleep(5.0)
                if stop_event.is_set():
                    break
                print(f"[Trainer] Memory recovered ({ram_avail}MB available). Resuming training...", flush=True)

            try:
                _worker_id, payload, error = game_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if error is not None:
                print(f"self-play worker error: {error}", flush=True)
                continue
            result, experiences, sans = payload
            replay.extend(experiences)
            completed_games += 1
            rated_side = WHITE if completed_games % 2 else BLACK
            opponent_elo = elo
            elo, elo_delta = update_rating_from_result(elo, opponent_elo, result, rated_side)
            iteration += 1 if completed_games % max(1, games_per_generation) == 0 else 0
            generation_value.value = iteration
            updates = 0
            if len(replay) > 2048:
                for _ in range(max(1, int(updates_per_game))):
                    train_updates += 1
                    updates += 1
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
                    teacher.update(model)
                    if train_updates % 10 == 0:
                        _copy_state_to_shared(teacher.teacher, shared_state, state_lock)
                        weights_updated.set()
            else:
                metrics = {"loss": 0.0, "lr": optimizer.param_groups[0]["lr"], "replay_size": float(len(replay))}
            metrics["elo"] = float(elo)
            metrics["elo_delta"] = float(elo_delta)
            print(
                f"game={completed_games} generation={iteration} replay={len(replay)} "
                f"updates={updates} loss={metrics.get('loss', 0.0):.4f} lr={metrics.get('lr', 0.0):.6g} "
                f"elo={elo:.1f} elo_delta={elo_delta:+.1f} rated_side={'white' if rated_side == WHITE else 'black'} result={result}",
                flush=True,
            )
            _append_training_game_history(
                completed_games,
                iteration,
                result,
                sans,
                elo,
                elo_delta,
                rated_side,
                len(replay),
                train_updates,
                metrics,
            )
            Path(replay_path).parent.mkdir(parents=True, exist_ok=True)
            replay.save(replay_path)
            checkpoint_manager.save(model, iteration, elo, optimizer.state_dict(), metrics)
            last_checkpoint = time.monotonic()

            # Explicitly clear variables from the current game/iteration to prevent leaks
            try:
                del payload
                del experiences
                del sans
                del metrics
            except NameError:
                pass
            gc.collect()
    finally:
        Path(replay_path).parent.mkdir(parents=True, exist_ok=True)
        replay.save(replay_path)
        checkpoint_manager.save(
            model,
            iteration,
            elo,
            optimizer.state_dict(),
            {"interrupted": 1.0, "step": float(train_updates), "elo": float(elo)},
        )


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


def _collect_eval_requests_nonblocking(request_queue, gpu_batch_size: int, wait_seconds: float, stop_event, active_workers=None):
    try:
        item = request_queue.get(timeout=0.05)
    except queue.Empty:
        return []
    if item[0] == "stop":
        stop_event.set()
        return []
    requests = [item]
    positions = len(item[3])
    deadline = time.monotonic() + wait_seconds
    while not stop_event.is_set():
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
            stop_event.set()
            break
        requests.append(item)
        positions += len(item[3])
    return requests


def _configure_cuda_process(device: str, torch_module, memory_fraction: float = 0.70) -> None:
    try:
        torch_module.set_num_threads(1)
        torch_module.set_num_interop_threads(1)
    except Exception:
        pass
    if device != "cuda":
        return
    torch_module.cuda.set_per_process_memory_fraction(max(0.10, min(0.95, memory_fraction)))
    torch_module.backends.cuda.matmul.allow_tf32 = True
    torch_module.backends.cudnn.benchmark = True
    torch_module.backends.cudnn.allow_tf32 = True
    torch_module.set_float32_matmul_precision("high")
    if hasattr(torch_module.backends.cuda, "enable_flash_sdp"):
        torch_module.backends.cuda.enable_flash_sdp(True)
        torch_module.backends.cuda.enable_mem_efficient_sdp(True)
        torch_module.backends.cuda.enable_math_sdp(True)


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
    last_time = time.monotonic()
    last_games = 0
    last_positions = 0
    while not stop_event.is_set():
        time.sleep(30.0)
        now = time.monotonic()
        elapsed = max(1e-6, now - last_time)
        games = int(games_completed.value)
        positions = int(positions_evaluated.value)
        batches = int(eval_batches.value)
        total_batch_positions = int(batch_positions.value)
        games_per_minute = (games - last_games) * 60.0 / elapsed
        positions_per_second = (positions - last_positions) / elapsed
        average_batch = total_batch_positions / max(1, batches)
        gpu_util, vram_used, vram_total = _query_gpu_utilization()
        ram_used, ram_total, ram_available, swap_used, swap_total = _query_memory_utilization()
        warning_threshold = min(128.0, max(1.0, target_batch_size * 0.75))
        warnings = []
        if average_batch < warning_threshold:
            warnings.append(f"avg_batch_below_{warning_threshold:.0f}")
        if swap_total > 0 and swap_used / swap_total > 0.50:
            warnings.append("swap_above_50pct")
        if ram_available < 1024:
            warnings.append("ram_available_below_1GiB")
        warning = f" {' '.join(warnings)}" if warnings else ""
        line = (
            f"gpu_util={gpu_util:.0f}% vram={vram_used}/{vram_total}MiB "
            f"ram={ram_used}/{ram_total}MiB ram_avail={ram_available}MiB "
            f"swap={swap_used}/{swap_total}MiB "
            f"games_per_min={games_per_minute:.2f} positions_per_sec={positions_per_second:.1f} "
            f"avg_batch={average_batch:.1f}{warning}"
        )
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        last_time = now
        last_games = games
        last_positions = positions


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
        return 0, 0, 0, 0, 0


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
    result_queue,
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
            c_puct=1.5 if config.generation < 200 else 1.0,
            batch_size=config.batch_size,
            add_noise=True,
            resign_threshold=-1.0 if config.generation < 10 else config.resign_value,
        )
        result_queue.put((worker_id, play_game(mcts, config, rng), None))
    except BaseException:
        result_queue.put((worker_id, None, traceback.format_exc()))
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

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

    model = ZeroNet(ModelConfig(**model_config)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    wait_seconds = max_wait_ms / 1000.0

    while True:
        requests = _collect_eval_requests(request_queue, gpu_batch_size, wait_seconds, active_workers)
        if requests is None:
            return
        flat_boards = [board for _, _, _, boards in requests for board in boards]
        try:
            flat_results = model.evaluate_batch(flat_boards, device)
            error = None
        except BaseException:
            flat_results = []
            error = traceback.format_exc()

        offset = 0
        for _, worker_id, request_id, boards in requests:
            end = offset + len(boards)
            response_queues[worker_id].put((request_id, None if error else flat_results[offset:end], error))
            offset = end


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


def _adjudicate(values: list[float]) -> bool:
    if len(values) < 20:
        return False
    return max(values) - min(values) <= 0.05 and abs(sum(values) / len(values)) > 0.9


def _finalize(
    records: list[PositionRecord],
    result: str,
    rng: random.Random,
    augment: bool,
) -> list[Experience]:
    experiences = []
    for idx, record in enumerate(records):
        terminal = game_result_to_value(result, record.turn)
        bootstrap = _bootstrap_value(records, idx)
        reward_bonus = (
            AGGRESSION_WEIGHT * record.aggression_score
            + record.momentum_reward
            + record.panic_penalty
        )
        exp = Experience(
            fen=record.fen,
            policy=record.policy,
            value=terminal,
            td_value=bootstrap,
            wdl=terminal_wdl(terminal),
            priority=abs(record.root_value - (terminal + reward_bonus)) + 1e-3,
            value_prediction=record.root_value,
            reward_bonus=reward_bonus,
            aggression_score=record.aggression_score,
            momentum_reward=record.momentum_reward,
            panic_penalty=record.panic_penalty,
        )
        experiences.append(exp)
        if augment and rng.random() < 0.5:
            experiences.append(_flip_experience_horizontal(exp))
    return experiences


def _bootstrap_value(records: list[PositionRecord], idx: int, plies: int = 5) -> float:
    if not records:
        return game_result_to_value("1/2-1/2", WHITE)
    j = min(len(records) - 1, idx + plies)
    current_turn = records[idx].turn
    later_turn = records[j].turn
    later_value = records[j].root_value
    return later_value if later_turn == current_turn else opponent_value(later_value)


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


PIECE_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}


def _aggression_score(board: Board) -> float:
    opponent = BLACK if board.turn == WHITE else WHITE
    opponent_squares = [
        sq for sq, piece in enumerate(board.squares)
        if piece != EMPTY and color_of(piece) == opponent
    ]
    if not opponent_squares:
        return 0.0
    attacked = sum(1 for sq in opponent_squares if board.is_square_attacked(sq, board.turn))
    return attacked / len(opponent_squares)


def _captured_piece(board: Board, move: Move) -> str:
    if move.flags & EN_PASSANT:
        captured_sq = move.to_sq - 8 if board.turn == WHITE else move.to_sq + 8
        return board.squares[captured_sq]
    return board.squares[move.to_sq]


def _momentum_reward(board: Board, move: Move) -> float:
    captured = _captured_piece(board, move)
    if captured == EMPTY:
        return 0.0
    return MOMENTUM_REWARD if PIECE_VALUES[piece_type(captured)] >= 1 else 0.0


def _panic_penalty(board: Board, move: Move) -> float:
    captured = _captured_piece(board, move)
    if captured == EMPTY:
        return 0.0
    return PANIC_PENALTY if PIECE_VALUES[piece_type(captured)] >= 1 else 0.0


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


def _flip_uci_horizontal(uci: str) -> str:
    return _flip_square_name(uci[:2]) + _flip_square_name(uci[2:4]) + uci[4:]


def _flip_square_name(name: str) -> str:
    sq = parse_square(name)
    return square_name(square(7 - file_of(sq), rank_of(sq)))


def evaluator_from_checkpoint(path: str | None, device: str):
    if not path:
        return UniformEvaluator()
    from .model import load_model

    return NetworkEvaluator(load_model(path, device), device)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate ZERO self-play games.")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replay-out", default="data/replay.pkl")
    parser.add_argument("--pgn-out")
    args = parser.parse_args(argv)

    evaluator = evaluator_from_checkpoint(args.checkpoint, args.device)
    replay = PrioritizedReplayBuffer(cold_path="data/replay_cold.sqlite3")
    from .pgn import export_pgn

    games = generate_parallel_games(evaluator, args.games, SelfPlayConfig(simulations=args.simulations))
    for game_idx, (result, experiences, sans) in enumerate(games):
        replay.extend(experiences)
        if args.pgn_out:
            pgn = export_pgn(
                sans,
                result,
                {
                    "Event": "ZERO self-play",
                    "Date": datetime.now(timezone.utc).strftime("%Y.%m.%d"),
                    "Round": str(game_idx + 1),
                    "White": "ZERO",
                    "Black": "ZERO",
                },
            )
            Path(args.pgn_out).parent.mkdir(parents=True, exist_ok=True)
            with Path(args.pgn_out).open("a", encoding="utf-8") as fh:
                fh.write(pgn + "\n\n")
        print({"game": game_idx + 1, "result": result, "positions": len(experiences)})
    replay.save(args.replay_out)


if __name__ == "__main__":  # pragma: no cover
    main()
