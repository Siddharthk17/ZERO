"""Continuous self-play training utilities and optimized optimization loops."""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .board import Board
from .constants import WHITE, BLACK
from .encoding import POLICY_SIZE, encode_board, encode_boards, encode_move_mask, move_to_policy_index
from .ewc import ElasticWeightConsolidation
from .model import ModelConfig, ZeroNet, load_model, parameter_count, save_model
from .move import Move
from .replay import Experience, PrioritizedReplayBuffer

@dataclass(slots=True)
class TrainConfig:
    """Hyperparameters for the training loop: batch size, learning rates, loss weights, and device."""
    batch_size: int = 2048  # 128GB RAM + Blackwell: massive batches
    initial_lr: float = 2e-3  # Higher LR for larger batches
    continuous_lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    td_lambda: float = 0.7
    value_weight: float = 1.0
    wdl_weight: float = 1.0
    ewc_weight: float = 1.0
    aux_weight: float = 0.3  # Increased auxiliary focus
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    log_path: str = "logs/training.log"

class ContinuousLRScheduler:
    """Manages cosine annealing rates decaying down to strict continuous baselines."""

    def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float = 1e-3, final_lr: float = 3e-5) -> None:
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.final_lr = final_lr

    def step(self, iteration: int) -> float:
        """Set the optimizer learning rate for the given iteration and return it."""
        lr = self.lr_at(iteration)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def lr_at(self, iteration: int) -> float:
        """Return the learning rate at the given iteration using cosine annealing."""
        if iteration >= 500:
            return self.final_lr
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
        target_keys = (
            "policy_loss", "value_loss", "wdl_loss", "ewc_loss", 
            "aux_loss", "loss", "policy_entropy", "value_error",
            "material_loss", "mobility_loss", "king_safety_loss"
        )
        for key in target_keys:
            if any(key in row for row in self.window):
                valid_vals = [row[key] for row in self.window if key in row]
                averages[f"avg_{key}_100"] = sum(valid_vals) / len(valid_vals)
        payload = {**metrics, **averages}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload

def make_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Instantiate a fused AdamW optimizer for Blackwell maximum throughput."""
    try:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.initial_lr,
            weight_decay=config.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.initial_lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

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
) -> tuple[dict[str, float], torch.amp.GradScaler]:
    """Execute a single, high-speed gradient backpropagation step.

    Returns the metrics dictionary and the ``GradScaler`` used (creating one if
    none was supplied so callers can reuse it across iterations).
    """
    model.train()
    beta = replay.anneal_beta(iteration)
    batch = replay.sample_with_weights(config.batch_size, beta=beta)

    device_type = "cuda" if str(config.device).startswith("cuda") else "cpu"
    use_pinned = device_type == "cuda"

    boards = [Board.from_fen(exp.fen) for exp in batch.experiences]
    legal_moves = [board.legal_moves() for board in boards]

    x = encode_boards(boards, device="cpu")
    mask = torch.stack([encode_move_mask(legal, board, device="cpu") for board, legal in zip(boards, legal_moves, strict=True)])
    policy_target = torch.stack([_policy_from_exp(board, exp, legal) for board, exp, legal in zip(boards, batch.experiences, legal_moves, strict=True)])

    value_targets = []
    wdl_targets = []
    material_targets = []
    mobility_targets = []
    king_targets = []
    for exp, board, legal in zip(batch.experiences, boards, legal_moves, strict=True):
        raw_target = td_blended_target(exp.value, exp.td_value, config.td_lambda) + exp.reward_bonus
        value_targets.append(max(-31.0, min(1.0, raw_target)))
        wdl_targets.append(exp.wdl)
        material, mobility, king_safety = _auxiliary_targets(board, legal)
        material_targets.append(material)
        mobility_targets.append(mobility)
        king_targets.append(king_safety)

    value_target = torch.tensor(value_targets, dtype=torch.float32)
    wdl_target = torch.tensor(wdl_targets, dtype=torch.float32)
    sample_weights = torch.tensor(batch.weights, dtype=torch.float32)
    material_target = torch.tensor(material_targets, dtype=torch.float32)
    mobility_target = torch.tensor(mobility_targets, dtype=torch.float32)
    king_target = torch.tensor(king_targets, dtype=torch.float32)

    if use_pinned:
        x = x.pin_memory().to(config.device, non_blocking=True)
        mask = mask.pin_memory().to(config.device, non_blocking=True)
        policy_target = policy_target.pin_memory().to(config.device, non_blocking=True)
        scalar_transfer = [t.pin_memory().to(config.device, non_blocking=True) for t in
                           (value_target, wdl_target, sample_weights, material_target, mobility_target, king_target)]
        value_target, wdl_target, sample_weights, material_target, mobility_target, king_target = scalar_transfer
    else:
        x, mask, policy_target = x.to(config.device), mask.to(config.device), policy_target.to(config.device)
        value_target = value_target.to(config.device)
        wdl_target = wdl_target.to(config.device)
        sample_weights = sample_weights.to(config.device)
        material_target = material_target.to(config.device)
        mobility_target = mobility_target.to(config.device)
        king_target = king_target.to(config.device)

    optimizer.zero_grad(set_to_none=True)
    device_type = "cuda" if str(config.device).startswith("cuda") else "cpu"
    use_amp = config.mixed_precision and device_type == "cuda"
    
    is_bf16 = device_type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if (is_bf16 or device_type == "cpu") else torch.float16
    
    if scaler is None:
        # Only fp16 autocast requires gradient scaling; bf16 has fp32 exponent range.
        needs_scaler = use_amp and amp_dtype == torch.float16
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=needs_scaler)

    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
        out = model(x, mask, return_dict=True)
        log_policy = F.log_softmax(out["masked_policy_logits"], dim=-1)
        policy_loss_vec = -(policy_target * log_policy).sum(dim=-1)
        
        value_pred = out["value"].squeeze(-1)
        value_loss_vec = F.mse_loss(value_pred, value_target, reduction="none")
        wdl_loss_vec = -(wdl_target * F.log_softmax(out["wdl_logits"], dim=-1)).sum(dim=-1)
        
        # Calculate individual auxiliary target losses [1]
        mat_loss_vec = F.mse_loss(out["material"], material_target, reduction="none")
        mob_loss_vec = F.mse_loss(out["mobility"], mobility_target, reduction="none")
        king_loss_vec = F.mse_loss(out["king_safety"], king_target, reduction="none")
        aux_loss_vec = mat_loss_vec + mob_loss_vec + king_loss_vec
        
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

        # Advanced Telemetry: Policy Entropy and Value Prediction Error
        with torch.no_grad():
            probs = torch.softmax(out["masked_policy_logits"], dim=-1)
            # Clip probabilities to prevent log(0)
            clipped_probs = torch.clamp(probs, min=1e-9)
            policy_entropy = -(probs * torch.log(clipped_probs)).sum(dim=-1).mean()
            value_error = torch.abs(value_pred - value_target).mean()

    loss_val = float(loss.detach().cpu())
    grad_norm_before = torch.zeros((), device=config.device)

    if not math.isfinite(loss_val) or loss_val > 2000.0:
        import warnings
        warnings.warn(
            f"Loss sanity check failed: loss = {loss_val:.4f}. Skipping optimizer step and replay buffer priority update to prevent weight corruption.",
            RuntimeWarning,
            stacklevel=2,
        )
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
        "loss": float(loss_val),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "wdl_loss": float(wdl_loss.detach().cpu()),
        "ewc_loss": float(ewc_loss.detach().cpu()),
        "aux_loss": float(aux_loss.detach().cpu()),
        "material_loss": float(mat_loss_vec.mean().detach().cpu()),
        "mobility_loss": float(mob_loss_vec.mean().detach().cpu()),
        "king_safety_loss": float(king_loss_vec.mean().detach().cpu()),
        "policy_entropy": float(policy_entropy.cpu()),
        "value_error": float(value_error.cpu()),
        "grad_norm": float(min(float(grad_norm_before.detach().cpu()), config.grad_clip)),
        "lr": optimizer.param_groups[0]["lr"],
        "replay_size": float(len(replay)),
        "beta": float(beta),
    }
    if logger is not None:
        metrics = logger.log(metrics)
    return metrics, scaler

def td_blended_target(outcome: float, bootstrap: float, lambda_td: float = 0.7) -> float:
    """Weight the terminal result and the bootstrapped TD reward prediction."""
    return lambda_td * outcome + (1.0 - lambda_td) * bootstrap

def _policy_from_exp(board: Board, exp: Experience, legal: list[Move] | None = None) -> torch.Tensor:
    """Format experience probabilities into the target policy vector shape (CPU tensor)."""
    target = torch.zeros(POLICY_SIZE, dtype=torch.float32)
    total = sum(exp.policy.values())
    if total <= 0:
        if legal is None:
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

_PIECE_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "p": -1, "n": -3, "b": -3, "r": -5, "q": -9}
_MAX_MATERIAL = 39.0
_MAX_MOBILITY = 218.0

def _auxiliary_targets(board: Board, legal: list[Move] | None = None) -> tuple[float, float, float]:
    """Calculate material balance, move mobility, and king safety auxiliary targets."""
    own = opp = 0
    squares = board.squares
    turn = board.turn

    for piece in squares:
        if piece == "." or piece == "K" or piece == "k":
            continue
        val = _PIECE_VALUES[piece.upper()]
        if (piece.isupper() and turn == WHITE) or (piece.islower() and turn == BLACK):
            own += val
        else:
            opp += val

    material = (own - opp) / _MAX_MATERIAL
    move_count = len(legal) if legal is not None else len(board.legal_moves())
    mobility = move_count / _MAX_MOBILITY
    king_safety = 0.0 if board.is_check(board.turn) else 1.0
    return material, mobility, king_safety

def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run one training step from a saved replay buffer."""
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
    metrics, _ = train_step(model, optimizer, replay, config, iteration=0, scheduler=scheduler, logger=TrainingLogger(config.log_path))
    save_model(args.out, model, optimizer=optimizer.state_dict(), metrics=metrics)
    print({"params": parameter_count(model), **metrics})

if __name__ == "__main__":
    main()