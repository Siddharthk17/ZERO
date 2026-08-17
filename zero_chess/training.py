"""Continuous ZERO-X optimization with policy and unified-WDL targets."""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .board import Board
from .encoding import POLICY_SIZE, encode_board, encode_move_mask, move_to_policy_index
from .model import NUM_VALUE_BINS, ZeroNet
from .replay import Experience, PrioritizedReplayBuffer


def compute_two_hot_target(
    targets: torch.Tensor, num_bins: int = NUM_VALUE_BINS, device: torch.device | None = None
) -> torch.Tensor:
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    target_device = targets.device if device is None else torch.device(device)
    target_dtype = targets.dtype if targets.is_floating_point() else torch.float32
    targets = targets.reshape(-1).to(device=target_device, dtype=target_dtype)
    bin_centers = torch.linspace(-1.0, 1.0, num_bins, device=target_device, dtype=target_dtype)
    bin_width = bin_centers[1] - bin_centers[0]

    clamped = targets.clamp(-1.0, 1.0)
    indices = ((clamped + 1.0) / bin_width).clamp(0, num_bins - 2).long()

    left_bin = bin_centers[indices]
    right_weight = (clamped - left_bin) / bin_width
    left_weight = 1.0 - right_weight

    two_hot = torch.zeros((targets.shape[0], num_bins), device=target_device, dtype=target_dtype)
    two_hot.scatter_add_(1, indices.unsqueeze(1), left_weight.unsqueeze(1))
    two_hot.scatter_add_(1, (indices + 1).unsqueeze(1), right_weight.unsqueeze(1))

    return two_hot


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 1024
    initial_lr: float = 2e-3
    continuous_lr: float = 1e-5
    total_steps: int = 600_000
    warmup_steps: int = 3_000
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    log_path: str = "logs/training.log"
    material_loss_weight: float = 0.05
    moves_left_loss_weight: float = 0.05
    # The opponent-response head is retained for research experiments but is
    # disabled in the canonical target because its label depends on an
    # unconditioned sampled move. Enable only with an explicitly defined label
    # producer.
    opponent_policy_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "initial_lr",
            "continuous_lr",
            "weight_decay",
            "grad_clip",
            "material_loss_weight",
            "moves_left_loss_weight",
            "opponent_policy_loss_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.batch_size <= 0 or self.total_steps <= 0 or self.warmup_steps < 0:
            raise ValueError("training step counts and batch_size must be valid")
        if self.opponent_policy_loss_weight > 0.0:
            raise ValueError("opponent_policy_loss_weight is disabled until the target stores the conditioning move")


class ContinuousLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float = 2e-3,
        final_lr: float = 1e-5,
        total_steps: int = 600_000,
        warmup_steps: int = 3_000,
    ) -> None:
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def lr_at(self, step: int) -> float:
        step = max(0, int(step))
        if step < self.warmup_steps:
            return self.initial_lr * step / max(1, self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        return self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * (1.0 + math.cos(math.pi * progress))

    def step(self, step: int) -> float:
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


class TrainingLogger:
    def __init__(self, path: str | Path, max_bytes: int = 256 * 1024 * 1024) -> None:
        self.path = Path(path)
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.window: deque[dict[str, float]] = deque(maxlen=100)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)

    def log(self, metrics: dict[str, float]) -> dict[str, float]:
        self.window.append(metrics)
        averages = {
            f"avg_{key}_100": sum(row[key] for row in self.window if key in row)
            / sum(key in row for row in self.window)
            for key in (
                "loss",
                "policy_loss",
                "value_loss",
                "wdl_loss",
                "material_loss",
                "moves_left_loss",
                "opponent_policy_loss",
                "policy_target_weight",
                "value_error",
            )
            if any(key in row for row in self.window)
        }
        payload = {**metrics, **averages}
        line = json.dumps(payload, sort_keys=True) + "\n"
        if self.path.exists() and self.path.stat().st_size + len(line.encode("utf-8")) > self.max_bytes:
            self.close()
            archive = self.path.with_name(self.path.name + ".1")
            try:
                archive.unlink()
            except FileNotFoundError:
                pass
            self.path.replace(archive)
            self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        self._stream.write(line)
        return payload

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass


def make_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    options = dict(lr=config.initial_lr, weight_decay=config.weight_decay, betas=(0.9, 0.95))
    if str(config.device).startswith("cuda"):
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **options)
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(model.parameters(), **options)


# Board/FEN encoding is Python-heavy and competes with Rayon and LibTorch. A
# bounded pool avoids oversubscribing the workstation while retaining overlap
# with the native self-play worker.
_ENCODING_WORKERS = min(8, max(1, os.cpu_count() or 1))
_ENCODING_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=_ENCODING_WORKERS,
    thread_name_prefix="zero-encoding",
)


def _encode_experience(exp: Experience) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one experience into its input planes and legal-move mask.

    Repetition counts are read from the Experience fields directly instead of
    re-counting occurrences on freshly FEN-parsed boards, whose hash-history
    would never reflect the sampled game's prior positions.
    """
    board = Board.from_fen(exp.fen)
    history = [Board.from_fen(fen) for fen in exp.history_fens]
    legal = board.legal_moves()

    planes = encode_board(board, history=history, device="cpu")
    if exp.repetitions >= 2:
        planes[12].fill_(1.0)
    if exp.repetitions >= 3:
        planes[13].fill_(1.0)
    for index, count in enumerate(exp.history_repetitions):
        base = (index + 1) * 14
        if count >= 2:
            planes[base + 12].fill_(1.0)
        if count >= 3:
            planes[base + 13].fill_(1.0)

    mask = encode_move_mask(legal, board, device="cpu")
    return planes, mask


def _encode_batch(experiences: list[Experience]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch through Rust when the native extension is available."""
    try:
        from .rust_bridge import _engine

        native = _engine()
        encode_native = getattr(native, "encode_training_batch")
    except (AttributeError, ImportError, OSError, RuntimeError):
        encoded = list(_ENCODING_EXECUTOR.map(_encode_experience, experiences))
        return torch.stack([row[0] for row in encoded]), torch.stack([row[1] for row in encoded])

    input_bytes, mask_bytes = encode_native(
        [exp.fen for exp in experiences],
        [list(exp.history_fens) for exp in experiences],
        [exp.repetitions for exp in experiences],
        [list(exp.history_repetitions) for exp in experiences],
    )
    input_tensor = torch.frombuffer(input_bytes, dtype=torch.float32).clone()
    mask_tensor = torch.frombuffer(mask_bytes, dtype=torch.uint8).clone()
    expected_inputs = len(experiences) * 121 * 8 * 8
    expected_mask = len(experiences) * POLICY_SIZE
    if input_tensor.numel() != expected_inputs or mask_tensor.numel() != expected_mask:
        raise RuntimeError("native training encoder returned an invalid buffer size")
    return (
        input_tensor.reshape(len(experiences), 121, 8, 8),
        mask_tensor.reshape(len(experiences), POLICY_SIZE).to(dtype=torch.float32),
    )


def _policy_from_exp(exp: Experience) -> torch.Tensor:
    target = torch.zeros(POLICY_SIZE, dtype=torch.float32)
    for index, probability in exp.policy.items():
        if 0 <= index < POLICY_SIZE:
            target[index] = max(0.0, float(probability))
    total = target.sum()
    if total > 0.0:
        target /= total
    else:
        board = Board.from_fen(exp.fen)
        legal = board.legal_moves()
        if legal:
            probability = 1.0 / len(legal)
            for move in legal:
                target[move_to_policy_index(board, move)] = probability
    return target


def _opponent_move_mask(exp: Experience) -> torch.Tensor:
    mask = torch.zeros(POLICY_SIZE, dtype=torch.float32)
    legal_indices = exp.opponent_legal_policy or tuple(exp.opponent_policy or ())
    if not legal_indices:
        return mask
    for index in legal_indices:
        if 0 <= index < POLICY_SIZE:
            mask[index] = 1.0
    return mask


_HORIZONTAL_POLICY_PLANE_MAP = (
    0,
    7,
    6,
    5,
    4,
    3,
    2,
    1,
    7,
    6,
    5,
    4,
    3,
    2,
    1,
    0,
)


def _build_policy_flip_indices() -> torch.Tensor:
    destinations = torch.empty(POLICY_SIZE, dtype=torch.long)
    for plane in range(73):
        if plane < 56:
            reflected_plane = _HORIZONTAL_POLICY_PLANE_MAP[plane // 7] * 7 + plane % 7
        elif plane < 64:
            reflected_plane = 56 + _HORIZONTAL_POLICY_PLANE_MAP[8 + plane - 56]
        else:
            promotion, direction = divmod(plane - 64, 3)
            reflected_plane = 64 + promotion * 3 + (2 - direction)
        for square in range(64):
            destinations[plane * 64 + square] = reflected_plane * 64 + (square & ~7) + (7 - (square & 7))
    return destinations


_POLICY_FLIP_INDICES = _build_policy_flip_indices()
_POLICY_FLIP_INDICES_BY_DEVICE: dict[torch.device, torch.Tensor] = {}


def _flip_policy_horizontally(policy: torch.Tensor) -> torch.Tensor:
    device = policy.device
    indices = _POLICY_FLIP_INDICES_BY_DEVICE.get(device)
    if indices is None or indices.device != device:
        indices = _POLICY_FLIP_INDICES.to(device)
        _POLICY_FLIP_INDICES_BY_DEVICE[device] = indices
    flipped = torch.empty_like(policy)
    flipped[:, indices] = policy
    return flipped


def _augment_batch(
    x: torch.Tensor,
    move_mask: torch.Tensor,
    policy_targets: torch.Tensor,
    opponent_policy_targets: torch.Tensor,
    opponent_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flip = torch.rand(x.shape[0]) < 0.5
    if flip.any():
        x_flipped = torch.flip(x[flip], dims=[-1]).clone()
        extra = 8 * 14
        x_flipped[:, [extra + 1, extra + 2]] = x_flipped[:, [extra + 2, extra + 1]].clone()
        x_flipped[:, [extra + 3, extra + 4]] = x_flipped[:, [extra + 4, extra + 3]].clone()
        x = x.clone()
        x[flip] = x_flipped

        move_mask = move_mask.clone()
        move_mask[flip] = _flip_policy_horizontally(move_mask[flip])
        policy_targets = policy_targets.clone()
        policy_targets[flip] = _flip_policy_horizontally(policy_targets[flip])
        opponent_policy_targets = opponent_policy_targets.clone()
        opponent_policy_targets[flip] = _flip_policy_horizontally(opponent_policy_targets[flip])
        opponent_mask = opponent_mask.clone()
        opponent_mask[flip] = _flip_policy_horizontally(opponent_mask[flip])
    return x, move_mask, policy_targets, opponent_policy_targets, opponent_mask


def _weighted_mean(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (losses * weights).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)


def train_step(
    model: ZeroNet,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplayBuffer,
    config: TrainConfig,
    iteration: int = 0,
    scheduler: ContinuousLRScheduler | None = None,
    logger: TrainingLogger | None = None,
) -> dict[str, float]:
    model.train()
    beta = replay.anneal_beta(iteration, config.total_steps)
    batch = replay.sample_with_weights(config.batch_size, beta=beta)
    x, mask = _encode_batch(batch.experiences)
    policy_targets = torch.stack([_policy_from_exp(exp) for exp in batch.experiences])
    opponent_mask = torch.stack([_opponent_move_mask(exp) for exp in batch.experiences])
    wdl_targets = torch.tensor([exp.wdl for exp in batch.experiences], dtype=torch.float32)
    q_mcts_targets = torch.tensor([[exp.q_mcts] for exp in batch.experiences], dtype=torch.float32)
    z_terminal_targets = wdl_targets[:, 0:1] - wdl_targets[:, 2:3]
    terminal_targets = torch.tensor(
        [exp.target_kind in {"terminal", "adjudicated"} for exp in batch.experiences], dtype=torch.bool
    ).unsqueeze(-1)
    hybrid_value_targets = torch.where(
        terminal_targets,
        0.5 * z_terminal_targets + 0.5 * q_mcts_targets,
        q_mcts_targets,
    )
    material_targets = torch.tensor([list(exp.material) for exp in batch.experiences], dtype=torch.float32)
    moves_left_targets = torch.tensor([[exp.moves_left] for exp in batch.experiences], dtype=torch.float32)
    opponent_policy_targets = torch.zeros((len(batch.experiences), POLICY_SIZE), dtype=torch.float32)
    opponent_policy_available = torch.zeros(len(batch.experiences), dtype=torch.float32)
    policy_target_weights = torch.tensor([exp.policy_weight for exp in batch.experiences], dtype=torch.float32)
    for row, exp in enumerate(batch.experiences):
        if exp.opponent_policy:
            for index, probability in exp.opponent_policy.items():
                opponent_policy_targets[row, index] = probability
            opponent_policy_available[row] = 1.0
    sample_weights = torch.tensor(batch.weights, dtype=torch.float32)

    x, mask, policy_targets, opponent_policy_targets, opponent_mask = _augment_batch(
        x, mask, policy_targets, opponent_policy_targets, opponent_mask
    )

    device = torch.device(config.device)
    model_device = next(model.parameters()).device
    if model_device.type != device.type or (device.index is not None and model_device.index != device.index):
        raise ValueError(f"model is on {model_device}, but training device is {device}")
    tensors = (
        x,
        mask,
        policy_targets,
        opponent_mask,
        wdl_targets,
        hybrid_value_targets,
        material_targets,
        moves_left_targets,
        opponent_policy_targets,
        opponent_policy_available,
        policy_target_weights,
        sample_weights,
        terminal_targets,
    )
    if device.type == "cuda":
        tensors = tuple(tensor.pin_memory().to(device, non_blocking=True) for tensor in tensors)
    else:
        tensors = tuple(tensor.to(device) for tensor in tensors)
    (
        x,
        mask,
        policy_targets,
        opponent_mask,
        wdl_targets,
        hybrid_value_targets,
        material_targets,
        moves_left_targets,
        opponent_policy_targets,
        opponent_policy_available,
        policy_target_weights,
        sample_weights,
        terminal_targets,
    ) = tensors

    two_hot_value_targets = compute_two_hot_target(
        hybrid_value_targets.squeeze(-1), num_bins=model.config.num_value_bins, device=device
    )

    optimizer.zero_grad(set_to_none=True)
    use_amp = config.mixed_precision and device.type == "cuda"
    if use_amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is unavailable; disable mixed_precision for this device")
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True) if use_amp else nullcontext()
    with autocast:
        out = model(x, mask, return_dict=True)
        policy_loss_vec = -(policy_targets * F.log_softmax(out["masked_policy_logits"], dim=-1)).sum(dim=-1)
        value_loss_vec = -(two_hot_value_targets * F.log_softmax(out["value_logits"], dim=-1)).sum(dim=-1)
        wdl_loss_vec = -(wdl_targets * F.log_softmax(out["wdl_logits"], dim=-1)).sum(dim=-1)
        material_loss_vec = (out["material"] - material_targets).pow(2).mean(dim=-1)
        moves_left_loss_vec = (out["moves_left"] - moves_left_targets).pow(2).squeeze(-1)

        masked_opponent_logits = out["opponent_policy_logits"].masked_fill(opponent_mask <= 0, -1e4)
        opponent_policy_loss_vec = -(opponent_policy_targets * F.log_softmax(masked_opponent_logits, dim=-1)).sum(
            dim=-1
        )

        policy_loss = _weighted_mean(policy_loss_vec, sample_weights * policy_target_weights)
        value_loss = _weighted_mean(value_loss_vec, sample_weights)
        wdl_loss = _weighted_mean(wdl_loss_vec, sample_weights * terminal_targets.squeeze(-1).float())
        material_loss = _weighted_mean(material_loss_vec, sample_weights)
        moves_left_loss = _weighted_mean(moves_left_loss_vec, sample_weights)
        opponent_denominator = (
            (sample_weights * opponent_policy_available * policy_target_weights)
            .sum()
            .clamp_min(torch.finfo(sample_weights.dtype).eps)
        )
        opponent_policy_loss = (
            opponent_policy_loss_vec * sample_weights * opponent_policy_available * policy_target_weights
        ).sum() / opponent_denominator

        loss = (
            policy_loss
            + 0.25 * value_loss
            + 0.50 * wdl_loss
            + config.material_loss_weight * material_loss
            + config.moves_left_loss_weight * moves_left_loss
            + config.opponent_policy_loss_weight * opponent_policy_loss
        )

    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite training loss at step {iteration}")
    if scheduler is not None:
        scheduler.step(iteration)
    loss.backward()
    try:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
    except RuntimeError as exc:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(f"non-finite gradients at step {iteration}") from exc
    optimizer.step()
    with torch.no_grad():
        value_pred = out["value"].detach().squeeze(-1)
        value_errors = (value_pred - hybrid_value_targets.squeeze(-1)).abs().float()
        probabilities = torch.softmax(out["masked_policy_logits"].detach(), dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-9))).sum(dim=-1).mean()
        policy_kl = (
            policy_targets * (torch.log(policy_targets.clamp_min(1e-9)) - torch.log(probabilities.clamp_min(1e-9)))
        ).sum(dim=-1)
        replay.update_priorities(
            batch.indices,
            (value_errors + policy_kl * policy_target_weights).float().cpu().tolist(),
            batch.generations,
        )

    metrics = {
        "step": float(iteration),
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "wdl_loss": float(wdl_loss.detach().cpu()),
        "material_loss": float(material_loss.detach().cpu()),
        "moves_left_loss": float(moves_left_loss.detach().cpu()),
        "opponent_policy_loss": float(opponent_policy_loss.detach().cpu()),
        "policy_target_weight": float(policy_target_weights.detach().mean().cpu()),
        "policy_entropy": float(entropy.detach().cpu()),
        "value_error": float((value_pred - hybrid_value_targets.squeeze(-1)).abs().mean().detach().cpu()),
        "grad_norm": float(min(float(grad_norm.detach().cpu()), config.grad_clip)),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "replay_size": float(len(replay)),
        "beta": float(beta),
        "terminal_target_fraction": float(terminal_targets.float().mean().detach().cpu()),
        "truncated_target_fraction": float((~terminal_targets).float().mean().detach().cpu()),
    }
    if logger is not None:
        metrics = logger.log(metrics)
    return metrics
