#!/usr/bin/env python3
"""Run the minimum end-to-end ZERO native/training readiness smoke test."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from zero_chess.model import ModelConfig, ZeroNet, export_torchscript
from zero_chess.replay import PrioritizedReplayBuffer
from zero_chess.rust_bridge import _engine, ingest_rust_batch
from zero_chess.training import TrainConfig, make_optimizer, train_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=1)
    parser.add_argument("--full-model", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    engine = _engine()
    device = torch.device(args.device)
    config = ModelConfig() if args.full_model else ModelConfig(channels=8, blocks=1, policy_channels=2)
    model = ZeroNet(config).to(device).eval()
    replay = PrioritizedReplayBuffer(hot_capacity=256)

    with tempfile.TemporaryDirectory(prefix="zero-preflight-") as directory:
        deployment = Path(directory) / "model.ts"
        export_torchscript(deployment, model, device)
        payload = engine.generate_self_play_batch_rust(
            str(deployment),
            num_games=args.games,
            simulations=args.simulations,
            batch_size=1,
            device=args.device,
            seed=0x5EED,
        )
        inserted = ingest_rust_batch(replay, payload)

    if inserted <= 0:
        raise SystemExit("native self-play returned no replay experiences")
    config = TrainConfig(batch_size=1, device=args.device, mixed_precision=False)
    metrics = train_step(model, make_optimizer(model, config), replay, config, iteration=1)
    print({
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "native_games": len(payload.get("games", [])),
        "replay_experiences": inserted,
        "training_loss": metrics["loss"],
        "status": "ready",
    })


if __name__ == "__main__":
    main()
