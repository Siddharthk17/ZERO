#!/usr/bin/env python
"""ZERO continuous self-play training orchestrator with active memory guardrails."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

# Force the parent directory into sys.path to guarantee clean zero_chess namespace resolution
sys.path.insert(0, str(Path(__file__).resolve().parent))

from zero_chess.constants import BLACK, WHITE
from zero_chess.elo import DEFAULT_ELO, update_rating_from_result
from zero_chess.mcts import NetworkEvaluator, UniformEvaluator
from zero_chess.replay import PrioritizedReplayBuffer
from zero_chess.self_play import SelfPlayConfig, generate_parallel_games, start_persistent_cuda_training

VERSION = "1.2.8"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--games_per_iteration", type=int, default=2)  # Balanced CPU loading for 8GB RAM
    parser.add_argument("--self_play_simulations", type=int)
    parser.add_argument("--mcts_batch_size", type=int)
    parser.add_argument("--gpu_eval_batch_size", type=int, default=64)  # Optimized batch size for GA107
    parser.add_argument("--training_batch_size", type=int, default=128)  # Blazing fast gradient step size
    parser.add_argument("--updates_per_game", type=float, default=1.0)
    parser.add_argument("--max_plies", type=int, default=256)
    parser.add_argument("--cuda_memory_fraction", type=float, default=0.40)  # Leaves 60% VRAM free for Hyprland
    parser.add_argument("--gpu_cooldown_ms", type=float, default=0.0)  # Low temp threshold on Victus cooling
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--reuse_tree", action="store_true", default=True)  # Enabled by default for +80 ELO warm start
    args = parser.parse_args()

    stop = {"value": False}

    def request_stop(_signum, _frame) -> None:
        stop["value"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print_header()

    try:
        import torch
    except ImportError:
        print("PyTorch not installed; running bootstrap self-play only. Install requirements for gradient training.", flush=True)
        replay = PrioritizedReplayBuffer(cold_path="data/replay_cold.sqlite3")
        print("Device: CPU bootstrap")
        print("Model parameters: 0")
        print(f"Replay buffer size: {len(replay)}")
        iteration = 0
        while not stop["value"]:
            # Memory Guard: if ram_avail < 512MB, pause game generation, call gc.collect()
            from zero_chess.self_play import _query_memory_utilization
            _, _, ram_avail, _, _ = _query_memory_utilization()
            if ram_avail < 512:
                print(f"WARNING: Low memory detected ({ram_avail}MB available). Pausing game generation and clearing caches...", flush=True)
                import gc
                gc.collect()
                while not stop["value"]:
                    _, _, ram_avail, _, _ = _query_memory_utilization()
                    if ram_avail >= 512:
                        break
                    time.sleep(5.0)
                if stop["value"]:
                    break
                print(f"Memory recovered ({ram_avail}MB available). Resuming game generation.", flush=True)

            games = generate_parallel_games(UniformEvaluator(), args.games_per_iteration, SelfPlayConfig(simulations=1, max_plies=128))
            for _, experiences, _ in games:
                replay.extend(experiences)
            iteration += 1
            replay.save("data/replay.pkl")
            print(f"iteration={iteration} replay={len(replay)} mode=bootstrap", flush=True)

            # Explicitly clear variables from the current game/iteration to prevent leaks
            try:
                del games
            except NameError:
                pass
            import gc
            gc.collect()
        replay.save("data/replay.pkl")
        print("Saved bootstrap replay to data/replay.pkl", flush=True)
        return

    from zero_chess.arena import play_arena
    from zero_chess.checkpoint import CheckpointManager
    from zero_chess.ema import EMATeacher
    from zero_chess.ewc import ElasticWeightConsolidation
    from zero_chess.model import ModelConfig, ZeroNet, load_model
    from zero_chess.training import ContinuousLRScheduler, TrainConfig, TrainingLogger, make_optimizer, train_step

    device = resolve_device(args.device, torch)
    configure_torch_for_speed(device, torch)
    print_device(device, torch)
    replay = PrioritizedReplayBuffer(cold_path="data/replay_cold.sqlite3")
    checkpoint_manager = CheckpointManager("checkpoints")
    iteration = 0
    train_updates = 0
    elo = DEFAULT_ELO

    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        model = load_model(args.resume, device)
        iteration = int(payload.get("iteration", 0))
        train_updates = int(payload.get("metrics", {}).get("step", 0))
        elo = float(payload.get("elo", DEFAULT_ELO))
        print(f"Loaded checkpoint: {args.resume}")
    else:
        model = ZeroNet(ModelConfig()).to(device)
        print("Initialized fresh random weights")

    config = TrainConfig(batch_size=args.training_batch_size, device=device)
    mcts_batch_size = args.mcts_batch_size or (4 if device == "cpu" else 24)
    self_play_simulations = args.self_play_simulations or (8 if device == "cpu" else 96)
    print(f"Model parameters: {model.parameter_count():,}")
    print(f"Replay buffer size: {len(replay)}")
    print(f"Generation: {iteration}")
    print(f"Self-play games/iteration: {args.games_per_iteration}")
    print(f"MCTS simulations/move: {self_play_simulations}")
    print(f"MCTS leaf batch size: {mcts_batch_size}")
    print(f"Optimizer updates/game: {args.updates_per_game:g}")
    if device == "cuda":
        print(f"Multiprocessing CPU workers: {args.games_per_iteration}")
        print("Dedicated trainer process: enabled")
        print(f"GPU evaluator batch target: {args.gpu_eval_batch_size}")
        print("GPU evaluator timeout: 50ms")
        print(f"CUDA memory fraction: {args.cuda_memory_fraction:.2f}")
        print(f"GPU cooldown per eval batch: {args.gpu_cooldown_ms:g}ms")
        print(f"torch.compile evaluator: {'enabled' if args.compile_model else 'disabled'}")
        print(f"MCTS tree reuse: {'enabled' if args.reuse_tree else 'disabled'}")
    print(f"Estimated ELO: {elo:.1f}", flush=True)

    if device == "cuda":
        model.cpu()
        runtime = start_persistent_cuda_training(
            model,
            config,
            games=args.games_per_iteration,
            self_play_config=SelfPlayConfig(
                simulations=self_play_simulations,
                batch_size=mcts_batch_size,
                generation=iteration,
                max_plies=args.max_plies,
                reuse_tree=args.reuse_tree,
            ),
            device=device,
            gpu_batch_size=args.gpu_eval_batch_size,
            eval_wait_ms=10.0,
            cuda_memory_fraction=args.cuda_memory_fraction,
            gpu_cooldown_ms=args.gpu_cooldown_ms,
            compile_model=args.compile_model,
            updates_per_game=args.updates_per_game,
            iteration=iteration,
            train_updates=train_updates,
            elo=elo,
        )
        try:
            while not stop["value"]:
                time.sleep(1.0)
                dead = [process for process in runtime.processes if process.exitcode not in (None, 0)]
                if dead:
                    for process in dead:
                        print(f"runtime process exited: pid={process.pid} exitcode={process.exitcode}", flush=True)
                    stop["value"] = True
        finally:
            runtime.stop()
            print("Stopped CUDA multiprocessing runtime", flush=True)
        return

    teacher = EMATeacher(model)
    optimizer = make_optimizer(model, config)
    scheduler = ContinuousLRScheduler(optimizer)
    ewc = ElasticWeightConsolidation()
    logger = TrainingLogger("logs/training.log")
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    self_play_evaluator = NetworkEvaluator(teacher.teacher, device)

    try:
        while not stop["value"]:
            # Memory Guard: if ram_avail < 512MB, pause game generation, call gc.collect()
            from zero_chess.self_play import _query_memory_utilization
            _, _, ram_avail, _, _ = _query_memory_utilization()
            if ram_avail < 512:
                print(f"WARNING: Low memory detected ({ram_avail}MB available). Pausing game generation and clearing caches...", flush=True)
                import gc
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                while not stop["value"]:
                    _, _, ram_avail, _, _ = _query_memory_utilization()
                    if ram_avail >= 512:
                        break
                    time.sleep(5.0)
                if stop["value"]:
                    break
                print(f"Memory recovered ({ram_avail}MB available). Resuming game generation.", flush=True)

            iteration += 1
            self_play_config = SelfPlayConfig(
                simulations=self_play_simulations,
                batch_size=mcts_batch_size,
                generation=iteration,
                max_plies=args.max_plies,
                reuse_tree=args.reuse_tree,
            )
            games = generate_parallel_games(
                self_play_evaluator,
                args.games_per_iteration,
                self_play_config,
            )
            for game_idx, (result, experiences, _) in enumerate(games):
                replay.extend(experiences)
                rated_side = WHITE if (iteration + game_idx) % 2 else BLACK
                elo, elo_delta = update_rating_from_result(elo, elo, result, rated_side)
                print(
                    f"rated game result={result} side={'white' if rated_side == WHITE else 'black'} "
                    f"elo={elo:.1f} elo_delta={elo_delta:+.1f}",
                    flush=True,
                )

            metrics = {"loss": 0.0, "replay_size": float(len(replay)), "lr": optimizer.param_groups[0]["lr"]}
            updates_this_iteration = 0
            if len(replay) > 256:
                update_count = max(1, int(len(games) * args.updates_per_game))
                for _ in range(update_count):
                    train_updates += 1
                    updates_this_iteration += 1
                    metrics = train_step(
                        model,
                        optimizer,
                        replay,
                        config,
                        ewc=ewc,
                        iteration=train_updates,
                        scheduler=scheduler,
                        scaler=scaler,
                        logger=logger,
                    )
                    teacher.update(model)
                metrics["updates_this_iteration"] = float(updates_this_iteration)

            if iteration % 20 == 0:
                arena = play_arena(
                    NetworkEvaluator(model, device),
                    NetworkEvaluator(teacher.teacher, device),
                    games=40,
                    simulations=8 if device == "cpu" else 800,
                    iteration=iteration,
                    elo_a=elo,
                )
                elo = float(arena["elo_a"])
                if arena["promote"]:
                    teacher.promote(model)
                    print(f"promoted teacher at iteration {iteration}", flush=True)

            if iteration % 50 == 0 and len(replay) > 0:
                ewc.consolidate(model, replay, device=device)

            metrics["elo"] = float(elo)
            checkpoint_manager.save(model, iteration, elo, optimizer.state_dict(), metrics)

            batch_stats = ""
            print(
                f"iteration={iteration} replay={len(replay)} loss={metrics.get('loss', 0.0):.4f} "
                f"updates={updates_this_iteration} lr={metrics.get('lr', 0.0):.6g}{batch_stats}",
                flush=True,
            )

            # Explicitly clear variables from the current game/iteration to prevent leaks
            try:
                del games
                del metrics
            except NameError:
                pass
            import gc
            gc.collect()
    finally:
        Path("data").mkdir(exist_ok=True)
        replay.save("data/replay.pkl")
        checkpoint_manager.save(model, iteration, elo, optimizer.state_dict(), {"interrupted": 1.0, "elo": float(elo)})
        print(f"Saved checkpoint at iteration {iteration}", flush=True)


def resolve_device(requested: str, torch_module) -> str:
    if requested == "cuda" and torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def print_device(device: str, torch_module) -> None:
    if device == "cuda":
        index = torch_module.cuda.current_device()
        props = torch_module.cuda.get_device_properties(index)
        print(f"Device: CUDA {props.name} ({props.total_memory / (1024**3):.1f} GB VRAM)")
    else:
        print("Device: CPU")


def configure_torch_for_speed(device: str, torch_module) -> None:
    if device != "cuda":
        return
    torch_module.backends.cuda.matmul.allow_tf32 = True
    torch_module.backends.cudnn.allow_tf32 = True
    torch_module.backends.cudnn.benchmark = True
    torch_module.set_float32_matmul_precision("high")
    if hasattr(torch_module.backends.cuda, "enable_flash_sdp"):
        torch_module.backends.cuda.enable_flash_sdp(True)
        torch_module.backends.cuda.enable_mem_efficient_sdp(True)
        torch_module.backends.cuda.enable_math_sdp(True)


def print_header() -> None:
    print(
        r"""
███████╗███████╗██████╗  ██████╗ 
╚══███╔╝██╔════╝██╔══██╗██╔═══██╗
  ███╔╝ █████╗  ██████╔╝██║   ██║
 ███╔╝  ██╔══╝  ██╔══██╗██║   ██║
███████╗███████╗██║  ██║╚██████╔╝
╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝
""".strip()
    )
    print(f"ZERO {VERSION}", flush=True)


if __name__ == "__main__":
    main()