"""31-day accepted-model self-play/training pipeline for ZERO-X."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import signal
import threading
import time
from pathlib import Path

import torch

from zero_chess.arena import accept_gate_result, gate_checkpoints
from zero_chess.checkpoint import CheckpointManager
from zero_chess.model import (
    ModelConfig,
    ZeroNet,
    export_torchscript,
    load_model,
    load_model_payload,
    model_hash,
    save_model,
)
from zero_chess.replay import PrioritizedReplayBuffer
from zero_chess.rust_bridge import (
    RustEngineUnavailableError,
    _engine,
    append_rust_game_history,
    generate_rust_self_play,
    ingest_rust_batch,
)
from zero_chess.training import (
    ContinuousLRScheduler,
    TrainConfig,
    TrainingLogger,
    make_optimizer,
    train_step,
)

RUN_STATE_VERSION = 1


def _load_run_state(path: Path, duration_days: float, fresh: bool) -> tuple[float, float]:
    if path.exists() and not fresh:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("version") != RUN_STATE_VERSION:
                raise ValueError("unsupported run-state version")
            started_at = float(state["started_at"])
            deadline = float(state["deadline"])
            if deadline > started_at:
                return started_at, deadline
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            print(f"[Run Warning] Unable to read {path}; starting a new run clock.", flush=True)
    started_at = time.time()
    deadline = started_at + duration_days * 86400.0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"version": RUN_STATE_VERSION, "started_at": started_at, "deadline": deadline}))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return started_at, deadline


def _valid_checkpoint_payload(path: Path, device: str):
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
        model = load_model_payload(payload, device)
        return payload, model
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError, pickle.PickleError) as exc:
        print(f"[Checkpoint Warning] Ignoring invalid checkpoint {path}: {exc}", flush=True)
        return None, None


def configure_blackwell(device: str) -> None:
    if torch.device(device).type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


class ContinuousSelfPlayWorker(threading.Thread):
    def __init__(self, replay: PrioritizedReplayBuffer, deployment_path: Path, history_path: Path, args) -> None:
        super().__init__(daemon=True)
        self.replay = replay
        self.deployment_path = deployment_path
        self.history_path = history_path
        self.args = args
        self._stop_requested = threading.Event()
        self.total_games = 0
        self.total_positions = 0
        self.lock = threading.Lock()
        self.consecutive_errors = 0
        self.fatal_error: str | None = None
        self.active_since: float | None = None
        self.last_progress = time.monotonic()

    def run(self) -> None:
        iteration = 0
        while not self._stop_requested.is_set():
            if not self.deployment_path.exists():
                time.sleep(0.1)
                continue
            iteration += 1
            try:
                with self.lock:
                    self.active_since = time.monotonic()
                payload = generate_rust_self_play(
                    str(self.deployment_path),
                    num_games=self.args.games_per_batch,
                    simulations=self.args.simulations,
                    batch_size=self.args.eval_batch_size,
                    device=self.args.device,
                    seed=(self.args.seed + iteration * 1000) % 1_000_000,
                    fast_search_weight=self.args.fast_search_weight,
                )
                if self._stop_requested.is_set():
                    break
                games = payload.get("games", [])
                if len(games) != self.args.games_per_batch:
                    raise RuntimeError(
                        f"native self-play returned {len(games)} games; expected {self.args.games_per_batch}"
                    )
                pos_added = ingest_rust_batch(self.replay, payload)
                if pos_added <= 0:
                    raise RuntimeError("native self-play returned no trainable experiences")
                append_rust_game_history(
                    payload,
                    self.history_path,
                    model_path=str(self.deployment_path),
                    batch_index=iteration,
                )
                with self.lock:
                    self.total_games += len(payload.get("games", []))
                    self.total_positions += pos_added
                    self.consecutive_errors = 0
                    self.last_progress = time.monotonic()
            except Exception as e:
                print(f"\n[SelfPlay Warning] {e}", flush=True)
                with self.lock:
                    self.consecutive_errors += 1
                    if self.consecutive_errors >= 5:
                        self.fatal_error = str(e)
                        return
                self._stop_requested.wait(1.0)
            finally:
                with self.lock:
                    self.active_since = None

    def stop(self) -> None:
        self._stop_requested.set()

    def stats(self) -> tuple[int, int]:
        with self.lock:
            return self.total_games, self.total_positions

    def health_error(self) -> str | None:
        with self.lock:
            if (
                self.fatal_error is None
                and self.active_since is not None
                and time.monotonic() - self.active_since > self.args.self_play_timeout
            ):
                self.fatal_error = f"native self-play stalled for more than {self.args.self_play_timeout:.0f} seconds"
            return self.fatal_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--days", type=float, default=31.0)
    parser.add_argument("--games-per-batch", "--games-per-iteration", dest="games_per_batch", type=int, default=128)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--fast-search-weight", type=float, default=0.25)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--training-batch-size", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--policy-channels", type=int, default=64)
    parser.add_argument("--target-replay-ratio", type=float, default=4.0)
    parser.add_argument("--warmup-experiences", type=int, default=10_000)
    parser.add_argument("--replay-capacity", type=int, default=4_000_000)
    parser.add_argument("--replay-path", type=Path, default=Path("checkpoints/zero_x/replay.pkl"))
    parser.add_argument("--history-path", type=Path, default=Path("data/training_games.jsonl"))
    parser.add_argument("--candidate-interval", type=int, default=5_000)
    parser.add_argument("--gate-games", type=int, default=80)
    parser.add_argument("--gate-simulations", type=int, default=64)
    parser.add_argument("--gate-device", default="cpu")
    parser.add_argument("--gate-batch-size", type=int, default=8)
    parser.add_argument("--gate-workers", type=int, default=0, help="0 selects min(games, 24) native gate workers")
    parser.add_argument("--gate-min-score", type=float, default=0.50)
    parser.add_argument("--gate-lower-bound", type=float, default=0.45)
    parser.add_argument("--replay-save-interval", type=int, default=5_000)
    parser.add_argument("--shutdown-timeout", type=float, default=300.0)
    parser.add_argument("--self-play-timeout", type=float, default=600.0)
    parser.add_argument("--run-state-path", type=Path, default=Path("checkpoints/zero_x/run_state.json"))
    parser.add_argument("--disable-gating", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="refuse to start when an accepted checkpoint or replay snapshot already exists",
    )
    parser.add_argument("--seed", type=int, default=0x5EED_5EED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.days <= 0.0:
        raise SystemExit("--days must be positive")
    if args.games_per_batch <= 0:
        raise SystemExit("--games-per-batch must be positive")
    if args.simulations <= 0:
        raise SystemExit("--simulations must be positive")
    if not 0.0 <= args.fast_search_weight <= 1.0:
        raise SystemExit("--fast-search-weight must be in 0..=1")
    if not 1 <= args.eval_batch_size <= 256:
        raise SystemExit("--eval-batch-size must be in 1..=256")
    if args.training_batch_size <= 0:
        raise SystemExit("--training-batch-size must be positive")
    if args.channels <= 0 or args.blocks <= 0 or args.policy_channels <= 0:
        raise SystemExit("model dimensions must be positive")
    if args.target_replay_ratio <= 0.0:
        raise SystemExit("--target-replay-ratio must be positive")
    if args.warmup_experiences <= 0:
        raise SystemExit("--warmup-experiences must be positive")
    if args.replay_capacity < args.warmup_experiences:
        raise SystemExit("--replay-capacity must be at least --warmup-experiences")
    if args.candidate_interval <= 0:
        raise SystemExit("--candidate-interval must be positive")
    if args.replay_save_interval <= 0:
        raise SystemExit("--replay-save-interval must be positive")
    if args.shutdown_timeout <= 0.0:
        raise SystemExit("--shutdown-timeout must be positive")
    if args.self_play_timeout <= 0.0:
        raise SystemExit("--self-play-timeout must be positive")
    if args.gate_games <= 0 or args.gate_games % 2:
        raise SystemExit("--gate-games must be a positive even number")
    if args.gate_simulations <= 0:
        raise SystemExit("--gate-simulations must be positive")
    if not 1 <= args.gate_batch_size <= 256:
        raise SystemExit("--gate-batch-size must be in 1..=256")
    if not 0 <= args.gate_workers <= 24:
        raise SystemExit("--gate-workers must be in 0..=24")
    if not 0.0 <= args.gate_min_score <= 1.0:
        raise SystemExit("--gate-min-score must be in 0..=1")
    if not 0.0 <= args.gate_lower_bound <= args.gate_min_score:
        raise SystemExit("--gate-lower-bound must be in 0..=--gate-min-score")

    configure_blackwell(args.device)
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    try:
        _engine()
    except (ImportError, OSError, RustEngineUnavailableError) as exc:
        raise SystemExit(f"Native Rust self-play extension is unavailable: {exc}") from exc
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA BF16 is unavailable; use --device cpu or a BF16-capable GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_dir = Path("checkpoints/zero_x")
    model_dir.mkdir(parents=True, exist_ok=True)
    deployment_path = model_dir / "latest_model.ts"
    accepted_checkpoint = model_dir / "accepted.pt"
    if args.fresh and (
        accepted_checkpoint.exists()
        or (model_dir / "latest.pt").exists()
        or args.replay_path.exists()
        or args.run_state_path.exists()
    ):
        raise SystemExit("--fresh requires a new checkpoint and replay path")

    _started_at, run_deadline = _load_run_state(args.run_state_path, args.days, args.fresh)
    if time.time() >= run_deadline:
        raise SystemExit("the persisted training run deadline has already been reached")

    model_config = ModelConfig(
        channels=args.channels,
        blocks=args.blocks,
        policy_channels=args.policy_channels,
    )
    train_config = TrainConfig(batch_size=args.training_batch_size, device=args.device)
    checkpoint_mgr = CheckpointManager(model_dir, keep_last=10, permanent_every=100_000)
    logger = TrainingLogger("logs/master_training.jsonl")

    step = 0
    resume_optimizer_state = None
    accepted_step = 0
    resume_path = args.resume
    checkpoint_payload = None
    model = None
    candidates = []
    if resume_path is not None:
        candidates.append(resume_path)
    if accepted_checkpoint not in candidates:
        candidates.append(accepted_checkpoint)
    if model_dir.joinpath("latest.pt") not in candidates:
        candidates.append(model_dir / "latest.pt")
    candidates.extend(sorted(model_dir.glob("zero_iter_*.pt"), reverse=True))
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload, loaded_model = _valid_checkpoint_payload(candidate, args.device)
        if payload is not None:
            checkpoint_payload = payload
            model = loaded_model
            resume_path = candidate
            break
    if model is None:
        model = ZeroNet(model_config).to(args.device)
    else:
        step = int(checkpoint_payload.get("metrics", {}).get("step", 0))
        resume_optimizer_state = checkpoint_payload.get("optimizer")

    optimizer = make_optimizer(model, train_config)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
    scheduler = ContinuousLRScheduler(
        optimizer,
        initial_lr=train_config.initial_lr,
        final_lr=train_config.continuous_lr,
        total_steps=train_config.total_steps,
    )

    try:
        replay = (
            PrioritizedReplayBuffer.load(args.replay_path, hot_capacity=args.replay_capacity)
            if args.replay_path.exists()
            else PrioritizedReplayBuffer(hot_capacity=args.replay_capacity)
        )
    except (OSError, ValueError, KeyError, TypeError, EOFError, pickle.PickleError) as exc:
        print(f"[Replay Warning] Unable to load {args.replay_path}: {exc}; starting empty.", flush=True)
        replay = PrioritizedReplayBuffer(hot_capacity=args.replay_capacity)
    replay.metadata["last_accepted_model_hash"] = model_hash(model)
    accepted_payload, accepted_model = _valid_checkpoint_payload(accepted_checkpoint, args.device)
    accepted_existed = accepted_payload is not None
    if accepted_existed:
        accepted_step = int(accepted_payload.get("metrics", {}).get("step", 0))
        del accepted_payload
        del accepted_model
    if not accepted_existed:
        save_model(
            accepted_checkpoint,
            model,
            iteration=step,
            optimizer=optimizer.state_dict(),
            metrics={"step": float(step), "accepted": 1.0},
        )
        accepted_step = step
    accepted_model = load_model(accepted_checkpoint, args.device)
    export_torchscript(deployment_path, accepted_model, args.device)
    del accepted_model

    worker = ContinuousSelfPlayWorker(replay, deployment_path, args.history_path, args)
    worker.start()
    model_is_accepted = (not accepted_existed) or resume_path == accepted_checkpoint

    stop = False

    def sig_handler(_s, _f):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    last_candidate_step = step
    last_replay_save_step = step
    total_samples_trained = 0

    print("==========================================================================", flush=True)
    print("  ZERO-X 31-Day Blackwell Workstation Training Launch", flush=True)
    print(
        f"  Device: {args.device} | Model: {args.blocks}b{args.channels} SE-ResNet "
        f"({model.parameter_count():,} params)",
        flush=True,
    )
    print(
        f"  RAM Replay Capacity: {args.replay_capacity:,} | SGD Batch Size: {args.training_batch_size}",
        flush=True,
    )
    print("==========================================================================", flush=True)

    try:
        while not stop:
            if worker.health_error() is not None:
                raise RuntimeError(f"self-play worker failed repeatedly: {worker.health_error()}")
            if not worker.is_alive() and worker.health_error() is None:
                raise RuntimeError("self-play worker exited unexpectedly")
            elapsed_days = max(0.0, (time.time() - _started_at) / 86400.0)
            if time.time() >= run_deadline:
                print("\n[COMPLETE] 31-day training target reached!", flush=True)
                break

            if len(replay) < args.warmup_experiences:
                print(
                    f"[Warming Replay Buffer] {len(replay):,}/{args.warmup_experiences:,} experiences...",
                    end="\r",
                    flush=True,
                )
                time.sleep(1.0)
                continue

            _, total_pos_gen = worker.stats()
            allowed_samples = total_pos_gen * args.target_replay_ratio
            if total_samples_trained >= allowed_samples:
                time.sleep(0.02)
                continue

            next_step = step + 1
            try:
                metrics = train_step(
                    model, optimizer, replay, train_config, iteration=next_step, scheduler=scheduler, logger=logger
                )
            except FloatingPointError as exc:
                print(f"\n[WARNING] {exc}. Skipping step {next_step} and recovering...", flush=True)
                continue
            step = next_step
            total_samples_trained += args.training_batch_size
            model_is_accepted = False

            if step - last_candidate_step >= args.candidate_interval:
                last_candidate_step = step
                candidate_path = model_dir / f"candidate_{step:07d}.pt"
                save_model(
                    candidate_path,
                    model,
                    iteration=step,
                    optimizer=optimizer.state_dict(),
                    metrics={**metrics, "candidate": 1.0},
                )
                accepted = args.disable_gating
                gate_metrics = {"score_fraction": 1.0, "score_low": 1.0}
                if not args.disable_gating:
                    try:
                        gate = gate_checkpoints(
                            candidate_path,
                            accepted_checkpoint,
                            games=args.gate_games,
                            simulations=args.gate_simulations,
                            device=args.gate_device,
                            seed=args.seed + step,
                            log_path="logs/gate.jsonl",
                            incumbent_deployment_path=(deployment_path if args.gate_device == args.device else None),
                            batch_size=args.gate_batch_size,
                            workers=args.gate_workers,
                        )
                        gate_metrics = gate.as_dict()
                        accepted = accept_gate_result(
                            gate,
                            min_score=args.gate_min_score,
                            min_lower_bound=args.gate_lower_bound,
                        )
                        print(
                            f"\n[GATE step={step}] score={gate.score_fraction:.3f} "
                            f"CI=[{gate.score_low:.3f},{gate.score_high:.3f}] "
                            f"elo_delta={gate.elo_difference:.1f} accepted={accepted}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"\n[GATE WARNING] {exc}; keeping incumbent.", flush=True)
                        accepted = False
                if accepted:
                    save_model(
                        accepted_checkpoint,
                        model,
                        iteration=step,
                        optimizer=optimizer.state_dict(),
                        metrics={**metrics, **gate_metrics, "accepted": 1.0},
                    )
                    export_torchscript(deployment_path, model, args.device)
                    replay.metadata["last_accepted_model_hash"] = model_hash(model)
                    model_is_accepted = True
                    accepted_step = step
                    checkpoint_mgr.save(
                        model,
                        step,
                        optimizer_state=optimizer.state_dict(),
                        metrics={**metrics, **gate_metrics, "accepted": 1.0},
                    )
                else:
                    accepted_payload = torch.load(accepted_checkpoint, map_location=args.device, weights_only=True)
                    model.load_state_dict(accepted_payload["model"], strict=True)
                    if accepted_payload.get("optimizer") is not None:
                        optimizer.load_state_dict(accepted_payload["optimizer"])
                    model_is_accepted = True
                    print(f"\n[GATE] Candidate step {step} rejected; incumbent retained.", flush=True)
                try:
                    candidate_path.unlink()
                except OSError:
                    pass

            if step - last_replay_save_step >= args.replay_save_interval:
                replay.save(args.replay_path)
                last_replay_save_step = step

            if step % 100 == 0:
                games_count, pos_count = worker.stats()
                print(
                    f"[Step {step:06d} | Day {elapsed_days:.2f}/{args.days:.1f}] "
                    f"Replay: {len(replay):,} | SelfPlay: {games_count:,} games ({pos_count:,} pos) | "
                    f"Loss: {metrics.get('loss', 0.0):.4f} | ValErr: {metrics.get('value_error', 0.0):.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}",
                    flush=True,
                )

    finally:
        worker.stop()
        worker.join(timeout=args.shutdown_timeout)
        if worker.is_alive():
            print(
                f"[SHUTDOWN] Self-play worker did not stop within {args.shutdown_timeout:.0f} seconds; "
                "continuing daemon shutdown.",
                flush=True,
            )
        print("\n[SHUTDOWN] Saving final model checkpoint...", flush=True)
        if not model_is_accepted:
            accepted_payload = torch.load(accepted_checkpoint, map_location=args.device, weights_only=True)
            model.load_state_dict(accepted_payload["model"], strict=True)
            if accepted_payload.get("optimizer") is not None:
                optimizer.load_state_dict(accepted_payload["optimizer"])
            accepted_step = int(accepted_payload.get("metrics", {}).get("step", accepted_step))
            replay.metadata["last_accepted_model_hash"] = model_hash(model)
        replay.save(args.replay_path)
        save_model(
            accepted_checkpoint,
            model,
            iteration=accepted_step,
            optimizer=optimizer.state_dict(),
            metrics={"step": float(accepted_step), "training_step": float(step), "accepted": 1.0},
        )
        checkpoint_mgr.save(
            model,
            accepted_step,
            optimizer_state=optimizer.state_dict(),
            metrics={"step": float(accepted_step), "training_step": float(step), "accepted": 1.0},
        )
        logger.close()


if __name__ == "__main__":
    main()
