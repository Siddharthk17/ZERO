"""Continuous self-play training utilities and optimized optimization loops."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .board import Board
from .constants import WHITE, BLACK
from .encoding import POLICY_SIZE, encode_board, encode_move_mask, move_to_policy_index
from .ema import update_ema_teacher
from .ewc import ElasticWeightConsolidation
from .model import ModelConfig, ZeroNet, load_model, parameter_count, save_model
from .move import Move
from .replay import Experience, PrioritizedReplayBuffer


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 128  # Highly stable, memory-safe batch size for 8GB system RAM
    initial_lr: float = 1e-3
    continuous_lr: float = 3e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    td_lambda: float = 0.7
    value_weight: float = 1.0
    wdl_weight: float = 1.0
    ewc_weight: float = 1.0
    aux_weight: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    log_path: str = "logs/training.log"


class ContinuousLRScheduler:
    """Manages cosine annealing rates decaying down to strict continuous baselines."""

    def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float = 1e-3, final_lr: float = 3e-5) -> None:
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=final_lr)

    def step(self, iteration: int) -> float:
        if iteration <= 500:
            for group in self.optimizer.param_groups:
                group["lr"] = self.initial_lr
            self.cosine.last_epoch = iteration - 1
            self.cosine.step()
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = self.final_lr
        return self.optimizer.param_groups[0]["lr"]

    def lr_at(self, iteration: int) -> float:
        if iteration >= 500:
            return self.final_lr
        import math
        return self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * (1.0 + math.cos(math.pi * iteration / 500.0))


class TrainingLogger:
    """Monitors losses and writes averages across sliding windows."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.window: deque[dict[str, float]] = deque(maxlen=100)

    def log(self, metrics: dict[str, float]) -> dict[str, float]:
        self.window.append(metrics)
        averages = {}
        for key in ("policy_loss", "value_loss", "wdl_loss", "ewc_loss", "aux_loss", "loss"):
            averages[f"avg_{key}_100"] = sum(row.get(key, 0.0) for row in self.window) / len(self.window)
        payload = {**metrics, **averages}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload


def make_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Instantiate a fused or regular AdamW optimizer based on hardware capabilities."""
    kwargs = {}
    if str(config.device).startswith("cuda"):
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay, **kwargs)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay)


def train_step(
    model: ZeroNet,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplayBuffer,
    config: TrainConfig,
    ewc: ElasticWeightConsolidation | None = None,
    iteration: int = 0,
    scheduler: ContinuousLRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    logger: TrainingLogger | None = None,
) -> dict[str, float]:
    """Execute a single, high-speed gradient backpropagation step."""
    model.train()
    beta = replay.anneal_beta(iteration)
    batch = replay.sample_with_weights(config.batch_size, beta=beta)
    
    tensors = []
    masks = []
    policy_targets = []
    value_targets = []
    wdl_targets = []
    material_targets = []
    mobility_targets = []
    king_targets = []

    for exp in batch.experiences:
        board = Board.from_fen(exp.fen)
        legal = board.legal_moves()
        tensors.append(encode_board(board, device=config.device))
        masks.append(encode_move_mask(legal, board, device=config.device))
        policy_targets.append(_policy_from_exp(board, exp, config.device))
        
        # CORRECTED: Clamp value targets to [-3.0, 1.0] matching the asymmetric draw-as-loss targets bounds [2]
        raw_target = td_blended_target(exp.value, exp.td_value, config.td_lambda) + exp.reward_bonus
        value_targets.append(max(-3.0, min(1.0, raw_target)))
        
        wdl_targets.append(exp.wdl)
        
        material, mobility, king_safety = _auxiliary_targets(board)
        material_targets.append(material)
        mobility_targets.append(mobility)
        king_targets.append(king_safety)

    x = torch.stack(tensors)
    mask = torch.stack(masks)
    policy_target = torch.stack(policy_targets)
    value_target = torch.tensor(value_targets, dtype=torch.float32, device=config.device)
    wdl_target = torch.tensor(wdl_targets, dtype=torch.float32, device=config.device)
    sample_weights = torch.tensor(batch.weights, dtype=torch.float32, device=config.device)
    material_target = torch.tensor(material_targets, dtype=torch.float32, device=config.device)
    mobility_target = torch.tensor(mobility_targets, dtype=torch.float32, device=config.device)
    king_target = torch.tensor(king_targets, dtype=torch.float32, device=config.device)

    optimizer.zero_grad(set_to_none=True)
    device_type = "cuda" if str(config.device).startswith("cuda") else "cpu"
    use_amp = config.mixed_precision and device_type == "cuda"
    
    # Safe try-except block to query driver precision support on multiple GPU architectures securely
    is_bf16 = False
    if device_type == "cuda":
        try:
            is_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        except Exception:
            pass
            
    amp_dtype = torch.bfloat16 if is_bf16 else torch.float16
    
    # Dynamic compatibility check to initialize the correct scaler based on the available PyTorch API version
    if scaler is None:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
        out = model(x, mask, return_dict=True)
        log_policy = F.log_softmax(out["masked_policy_logits"], dim=-1)
        policy_loss_vec = -(policy_target * log_policy).sum(dim=-1)
        
        value_pred = out["value"].squeeze(-1)
        value_loss_vec = F.mse_loss(value_pred, value_target, reduction="none")
        wdl_loss_vec = -(wdl_target * F.log_softmax(out["wdl_logits"], dim=-1)).sum(dim=-1)
        
        material_loss = F.mse_loss(out["material"], material_target, reduction="none")
        mobility_loss = F.mse_loss(out["mobility"], mobility_target, reduction="none")
        king_loss = F.mse_loss(out["king_safety"], king_target, reduction="none")
        aux_loss_vec = material_loss + mobility_loss + king_loss
        
        policy_loss = (policy_loss_vec * sample_weights).mean()
        value_loss = (value_loss_vec * sample_weights).mean()
        wdl_loss = (wdl_loss_vec * sample_weights).mean()
        aux_loss = (aux_loss_vec * sample_weights).mean()
        ewc_loss = ewc.loss(model) if ewc else torch.zeros((), device=config.device)
        
        loss = (
            policy_loss
            + config.value_weight * value_loss
            + config.wdl_weight * wdl_loss
            + config.ewc_weight * ewc_loss
            + config.aux_weight * aux_loss
        )

    loss_val = float(loss.detach().cpu())
    grad_norm_before = torch.zeros((), device=config.device)

    if loss_val > 100.0:
        import warnings
        warnings.warn(f"Loss sanity check failed: loss = {loss_val:.4f} > 100.0. Skipping optimizer step and replay buffer priority update to prevent weight corruption.")
        optimizer.zero_grad(set_to_none=True)
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step(iteration)

        with torch.no_grad():
            priorities = (value_pred.detach() - value_target).abs().cpu().tolist()
            replay.update_priorities(batch.indices, priorities)

    metrics = {
        "step": float(iteration),
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "wdl_loss": float(wdl_loss.detach().cpu()),
        "ewc_loss": float(ewc_loss.detach().cpu()),
        "aux_loss": float(aux_loss.detach().cpu()),
        "grad_norm": float(min(float(grad_norm_before.detach().cpu()), config.grad_clip)),
        "lr": optimizer.param_groups[0]["lr"],
        "replay_size": float(len(replay)),
        "beta": float(beta),
    }
    if logger is not None:
        metrics = logger.log(metrics)
    return metrics


def td_blended_target(outcome: float, bootstrap: float, lambda_td: float = 0.7) -> float:
    """Weight the terminal result and the bootstrapped TD reward prediction."""
    return lambda_td * outcome + (1.0 - lambda_td) * bootstrap


def _policy_from_exp(board: Board, exp: Experience, device: str) -> torch.Tensor:
    """Format experience probabilities into the target policy vector shape."""
    target = torch.zeros(POLICY_SIZE, dtype=torch.float32, device=device)
    total = sum(exp.policy.values())
    if total <= 0:
        legal = board.legal_moves()
        if legal:
            prob = 1.0 / len(legal)
            for move in legal:
                target[move_to_policy_index(board, move)] = prob
        return target
    for uci, prob in exp.policy.items():
        move = Move.from_uci(uci)
        target[move_to_policy_index(board, move)] = float(prob) / total
    return target


def _auxiliary_targets(board: Board) -> tuple[float, float, float]:
    """Calculate material balance, move mobility, and king safety auxiliary targets."""
    values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}
    own = opp = 0
    squares = board.squares
    turn = board.turn
    
    for piece in squares:
        if piece == "." or piece == "K" or piece == "k":
            continue
        val = values[piece.upper()]
        if (piece.isupper() and turn == WHITE) or (piece.islower() and turn == BLACK):
            own += val
        else:
            opp += val
            
    material = (own - opp) / 39.0
    mobility = len(board.legal_moves()) / 218.0
    king_safety = 0.0 if board.is_check(board.turn) else 1.0
    return material, mobility, king_safety


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one ZERO training step from a replay buffer.")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", default="checkpoints/zero.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    replay = PrioritizedReplayBuffer.load(args.replay)
    model = load_model(args.checkpoint, args.device) if args.checkpoint else ZeroNet(ModelConfig()).to(args.device)
    config = TrainConfig(batch_size=args.batch_size, device=args.device)
    optimizer = make_optimizer(model, config)
    scheduler = ContinuousLRScheduler(optimizer, config.initial_lr, config.continuous_lr)
    metrics = train_step(model, optimizer, replay, config, iteration=0, scheduler=scheduler, logger=TrainingLogger(config.log_path))
    save_model(args.out, model, optimizer=optimizer.state_dict(), metrics=metrics)
    print({"params": parameter_count(model), **metrics})


if __name__ == "__main__":  # pragma: no cover
    main()