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
from zero_chess.elo import DEFAULT_ELO, update_rating_with_reason
from zero_chess.mcts import NetworkEvaluator, UniformEvaluator
from zero_chess.replay import PrioritizedReplayBuffer
from zero_chess.self_play import (
    SelfPlayConfig,
    _append_training_game_history,
    generate_multiprocess_games,
    start_persistent_cuda_training,
)

VERSION = "1.2.8.7"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--games_per_iteration", type=int, default=16, help="Parallel game processes")
    parser.add_argument("--self_play_simulations", type=int, default=200)
    parser.add_argument("--mcts_batch_size", type=int, default=48)
    parser.add_argument("--training_batch_size", type=int, default=1024)
    parser.add_argument("--updates_per_game", type=float, default=20)
    parser.add_argument("--max_plies", type=int, default=512)
    parser.add_argument("--cuda_memory_fraction", type=float, default=0.90)
    parser.add_argument("--compile_model", action="store_true", default=False)
    parser.add_argument("--reuse_tree", action="store_true", default=True)
    parser.add_argument("--gpu_batch_size", type=int, default=192)
    parser.add_argument("--max_wait_ms", type=float, default=20.0)
    parser.add_argument("--channels", type=int, default=384)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--policy_channels", type=int, default=64)
    parser.add_argument("--resign_value", type=float, default=-0.99)
    parser.add_argument("--enable-resign", action="store_true", default=False, help="Enable resignation during self-play (default: disabled for full-game learning)")
    parser.add_argument("--temperature_moves", type=int, default=30)
    parser.add_argument("--opening_random_plies", type=int, default=8)
    parser.add_argument("--opening_random_prob", type=float, default=0.5)
    parser.add_argument("--max_iterations", type=int, default=0, help="Stop after this many iterations (0 = unlimited).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for self-play generation.")
    parser.add_argument("--persistent", action="store_true", default=False, help="Use persistent multiprocess workers instead of spawning per iteration.")
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
        from zero_chess.self_play import _query_memory_utilization, play_game
        from zero_chess.mcts import MCTS

        replay = PrioritizedReplayBuffer(cold_path="data/replay_cold.sqlite3")
        print("Device: CPU bootstrap")
        print("Model parameters: 0")
        print(f"Replay buffer size: {len(replay)}")
        iteration = 0
        while not stop["value"]:
            # Memory Guard: if ram_avail < 512MB, pause game generation, call gc.collect()
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

            bootstrap_rng = __import__("random").Random()
            bootstrap_games = []
            for _ in range(args.games_per_iteration):
                mcts = MCTS(UniformEvaluator(), batch_size=24, simulations=1)
                bootstrap_games.append(play_game(mcts, SelfPlayConfig(simulations=1, max_plies=128), bootstrap_rng))
                mcts.reset()
            games = bootstrap_games
            for game_idx, (result, experiences, sans, reason, meta) in enumerate(games):
                replay.extend(experiences)
                _append_training_game_history(
                    game_number=iteration * args.games_per_iteration + game_idx,
                    generation=iteration,
                    result=result,
                    moves_san=sans,
                    elo_after=0.0,
                    elo_delta=0.0,
                    rated_side=0,
                    replay_size=len(replay),
                    train_step=0,
                    metrics={},
                )
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
    from zero_chess.self_play import _query_memory_utilization
    from zero_chess.training import ContinuousLRScheduler, TrainConfig, TrainingLogger, make_optimizer, train_step

    device = resolve_device(args.device, torch)
    configure_torch_for_speed(device, torch)
    print_device(device, torch)
    iteration = 0
    train_updates = 0
    elo = DEFAULT_ELO
    payload = {}

    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        model = load_model(args.resume, device)
        iteration = int(payload.get("iteration", 0))
        train_updates = int(payload.get("metrics", {}).get("step", 0))
        elo = float(payload.get("elo", DEFAULT_ELO))
        print(f"Loaded checkpoint: {args.resume}")
    else:
        model = ZeroNet(ModelConfig(
            channels=args.channels,
            blocks=args.blocks,
            attention_heads=args.attention_heads,
            policy_channels=args.policy_channels,
        )).to(device)
        print("Initialized fresh random weights")

    config = TrainConfig(batch_size=args.training_batch_size, device=device)
    mcts_batch_size = args.mcts_batch_size
    self_play_simulations = args.self_play_simulations
    print(f"Model parameters: {model.parameter_count():,}")
    print(f"Generation: {iteration}")
    print(f"Self-play games/iteration: {args.games_per_iteration}")
    print(f"MCTS simulations/move: {self_play_simulations}")
    print(f"MCTS leaf batch size: {mcts_batch_size}")
    print(f"Optimizer updates/game: {args.updates_per_game:g}")
    if device == "cuda":
        print(f"Multiprocess workers (worker-side encoding): {args.games_per_iteration}")
        print(f"GPU evaluator batch size: {args.gpu_batch_size}")
        print(f"CUDA memory fraction: {args.cuda_memory_fraction:.2f}")
        print(f"MCTS tree reuse: {'enabled' if args.reuse_tree else 'disabled'}")
    print(f"Estimated ELO: {elo:.1f}", flush=True)

    if args.persistent:
        self_play_config = SelfPlayConfig(
            simulations=self_play_simulations,
            batch_size=mcts_batch_size,
            generation=iteration,
            max_plies=args.max_plies,
            reuse_tree=args.reuse_tree,
            resign_value=args.resign_value,
            disable_resign=not args.enable_resign,
            temperature_moves=args.temperature_moves,
            opening_random_plies=args.opening_random_plies,
            opening_random_prob=args.opening_random_prob,
        )
        runtime = start_persistent_cuda_training(
            model,
            config,
            games=args.games_per_iteration,
            self_play_config=self_play_config,
            device=device,
            gpu_batch_size=args.gpu_batch_size,
            eval_wait_ms=args.max_wait_ms,
            cuda_memory_fraction=args.cuda_memory_fraction,
            compile_model=args.compile_model,
            updates_per_game=args.updates_per_game,
            iteration=iteration,
            train_updates=train_updates,
            elo=elo,
        )
        print("Started persistent CUDA training runtime.", flush=True)
        with runtime:
            while not stop["value"]:
                if args.max_iterations > 0 and runtime.generation_value.value >= args.max_iterations:
                    print(f"Reached max_iterations={args.max_iterations}; stopping persistent runtime.", flush=True)
                    break
                time.sleep(2.0)
        return

    replay = PrioritizedReplayBuffer(cold_path="data/replay_cold.sqlite3")
    checkpoint_manager = CheckpointManager("checkpoints")
    print(f"Replay buffer size: {len(replay)}")

    teacher = EMATeacher(model)
    optimizer = make_optimizer(model, config)
    if "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
            print("Loaded optimizer state")
        except Exception as opt_exc:
            print(f"Could not load optimizer state: {opt_exc}", flush=True)
    scheduler = ContinuousLRScheduler(optimizer, config.initial_lr, config.continuous_lr)
    ewc = ElasticWeightConsolidation()
    logger = TrainingLogger("logs/training.log")
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    needs_scaler = device == "cuda" and not use_bf16
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

    last_checkpoint_iter = -1
    try:
        while not stop["value"]:
            # Memory Guard: if ram_avail < 512MB, pause game generation, call gc.collect()
            _, _, ram_avail, _, _ = _query_memory_utilization()
            if ram_avail < 512:
                print(f"WARNING: Low memory detected ({ram_avail}MB available). Pausing game generation...", flush=True)
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
                print(f"Memory recovered ({ram_avail}MB available). Resuming.", flush=True)

            iteration += 1
            self_play_config = SelfPlayConfig(
                simulations=self_play_simulations,
                batch_size=mcts_batch_size,
                generation=iteration,
                max_plies=args.max_plies,
                reuse_tree=args.reuse_tree,
                resign_value=args.resign_value,
                disable_resign=not args.enable_resign,
                temperature_moves=args.temperature_moves,
                opening_random_plies=args.opening_random_plies,
                opening_random_prob=args.opening_random_prob,
            )

            eval_model = teacher.teacher
            games = generate_multiprocess_games(
                eval_model,
                device=device,
                games=args.games_per_iteration,
                config=self_play_config,
                rng_seed=args.seed,
                gpu_batch_size=args.gpu_batch_size,
                max_wait_ms=args.max_wait_ms,
            )
            
            metrics = {"loss": 0.0, "replay_size": float(len(replay)), "lr": optimizer.param_groups[0]["lr"]}
            updates_this_iteration = 0
            if len(replay) > 256:
                update_count = max(1, int(len(games) * args.updates_per_game))
                for _ in range(update_count):
                    train_updates += 1
                    updates_this_iteration += 1
                    metrics, scaler = train_step(
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

            for game_idx, (result, experiences, sans, reason, meta) in enumerate(games):
                replay.extend(experiences)
                rated_side = WHITE if (iteration + game_idx) % 2 else BLACK
                elo, elo_delta = update_rating_with_reason(elo, result, rated_side, reason)
                print(
                    f"[GAME #{iteration:05d}-{game_idx}] result={result} side={'white' if rated_side == WHITE else 'black'} "
                    f"elo={elo:.1f} ({elo_delta:+.1f}) plies={len(sans)} reason={reason} duration={meta['duration']:.1f}s",
                    flush=True,
                )
                _append_training_game_history(
                    game_number=iteration * args.games_per_iteration + game_idx,
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

            if iteration % 20 == 0:
                arena = play_arena(
                    NetworkEvaluator(model, device),
                    NetworkEvaluator(teacher.teacher, device),
                    games=40,
                    simulations=800,
                    iteration=iteration,
                    elo_a=elo,
                )
                elo = float(arena["elo_a"])
                if arena["promote"]:
                    teacher.promote(model)
                    print(f"[ARENA] promoted teacher at iteration {iteration}", flush=True)

            if iteration % 50 == 0 and len(replay) > 0:
                ewc.consolidate(model, replay, device=device)

            if iteration != last_checkpoint_iter:
                metrics["elo"] = float(elo)
                checkpoint_manager.save(model, iteration, elo, optimizer.state_dict(), metrics)
                last_checkpoint_iter = iteration

            print(
                f"[ITER {iteration}] replay={len(replay):,} loss={metrics.get('loss', 0.0):.4f} "
                f"updates={updates_this_iteration} lr={metrics.get('lr', 0.0):.6g}",
                flush=True,
            )

            if args.max_iterations > 0 and iteration >= args.max_iterations:
                print(f"Reached max_iterations={args.max_iterations}; stopping gracefully.", flush=True)
                break

            try:
                del games, metrics
            except NameError:
                pass
            import gc
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        try:
            Path("data").mkdir(exist_ok=True)
            replay.save("data/replay.pkl")
            checkpoint_manager.save(model, iteration, elo, optimizer.state_dict(), {"interrupted": 1.0, "elo": float(elo)})
            print(f"[SAVE] Checkpoint at iteration {iteration}", flush=True)
        except Exception as save_exc:
            print(f"[SAVE] Error during emergency save: {save_exc}", flush=True)

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
    if torch_module.cuda.get_device_capability() >= (8, 0):
        try:
            torch_module._inductor.config.conv_benchmark = True
        except AttributeError:
            pass

def print_header() -> None:
    try:
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
    except UnicodeEncodeError:
        print("")
    print(f"ZERO {VERSION}", flush=True)

if __name__ == "__main__":
    main()