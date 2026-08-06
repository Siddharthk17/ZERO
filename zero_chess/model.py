"""Squeeze-and-Excitation ResNet for ZERO-X (Blackwell Tensor Core Optimized)."""

from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import POLICY_SIZE, encode_boards, encode_move_mask, move_to_policy_index

NUM_VALUE_BINS = 128


class LayerNorm2d(nn.Module):
    """Channel-first LayerNormalization for spatial feature maps.

    Replaces BatchNorm2d for RL self-play: inference batch sizes vary
    dynamically during MCTS (1..256) while training uses large batches, so
    BatchNorm running statistics drift and destabilise BF16 evaluation.
    LayerNorm has no running mean/variance state, giving identical behaviour
    across every batch size and numerical stability in bfloat16.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        u = x_float.mean(1, keepdim=True)
        s = (x_float - u).pow(2).mean(1, keepdim=True)
        x_norm = (x_float - u) / torch.sqrt(s + self.eps)
        return (self.weight[:, None, None] * x_norm + self.bias[:, None, None]).to(dtype=x.dtype)


@dataclass(slots=True)
class ModelConfig:
    input_channels: int = 121
    channels: int = 256
    blocks: int = 12
    policy_size: int = POLICY_SIZE
    se_reduction: int = 8
    policy_channels: int = 64
    num_value_bins: int = NUM_VALUE_BINS


class SwiGLUSqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.fc1 = nn.Linear(channels, hidden * 2)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        pooled = x.mean(dim=(2, 3))
        gate, val = self.fc1(pooled).chunk(2, dim=-1)
        swiglu = F.silu(gate) * val
        scale = torch.sigmoid(self.fc2(swiglu))
        return x * scale.view(batch, channels, 1, 1)


class ConvResidualBlock(nn.Module):
    """ConvNeXt-style residual block with LayerNorm2d and SwiGLU SE."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = LayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.se = SwiGLUSqueezeExcitation(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(F.silu(self.norm1(x)))
        y = self.conv2(F.silu(self.norm2(y)))
        return residual + self.se(y)


class ZeroNet(nn.Module):
    """Deep SE-ResNet with Categorical Value, WDL, Policy, and Aux Heads."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        channels = self.config.channels

        self.stem = nn.Sequential(
            nn.Conv2d(self.config.input_channels, channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.SiLU(),
        )
        self.tower = nn.Sequential(*[
            ConvResidualBlock(channels, reduction=self.config.se_reduction)
            for _ in range(self.config.blocks)
        ])

        # Policy Head (73 Planes)
        self.policy_head = nn.Sequential(
            LayerNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.config.policy_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(self.config.policy_channels, self.config.policy_size // 64, kernel_size=1),
            nn.Flatten(),
        )

        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_body = nn.Sequential(nn.Flatten(), nn.Linear(channels, 256), nn.SiLU())

        # Categorical Two-Hot Value Head
        self.categorical_value_head = nn.Linear(256, self.config.num_value_bins)

        # WDL Distribution Head
        self.wdl_head = nn.Linear(256, 3)

        # Material & Moves-Left Auxiliaries
        self.material_head = nn.Sequential(nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 2), nn.Sigmoid())
        self.moves_left_head = nn.Sequential(nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.opponent_policy_head = nn.Linear(256, self.config.policy_size)

        self.register_buffer("bin_centers", torch.linspace(-1.0, 1.0, self.config.num_value_bins))

    def forward(self, x: torch.Tensor, move_mask: torch.Tensor | None = None, return_dict: bool = True):
        features = self.tower(self.stem(x))
        policy_logits = self.policy_head(features)

        if move_mask is not None:
            masked_policy_logits = policy_logits.masked_fill(move_mask <= 0, -1e4)
        else:
            masked_policy_logits = policy_logits
        policy = torch.softmax(masked_policy_logits, dim=-1)

        value_features = self.value_body(self.value_pool(features))
        value_logits = self.categorical_value_head(value_features)
        value_probs = torch.softmax(value_logits, dim=-1)
        scalar_value = (value_probs * self.bin_centers).sum(dim=-1, keepdim=True)

        wdl_logits = self.wdl_head(value_features)
        wdl = torch.softmax(wdl_logits, dim=-1)

        material = self.material_head(value_features)
        moves_left = self.moves_left_head(value_features)
        opponent_policy_logits = self.opponent_policy_head(value_features)

        if not return_dict:
            return policy, scalar_value, wdl

        return {
            "policy_logits": policy_logits,
            "masked_policy_logits": masked_policy_logits,
            "policy": policy,
            "value_logits": value_logits,
            "value_probs": value_probs,
            "value": scalar_value,
            "scalar_value": scalar_value,
            "wdl_logits": wdl_logits,
            "wdl": wdl,
            "material": material,
            "moves_left": moves_left,
            "opponent_policy_logits": opponent_policy_logits,
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.inference_mode()
    def evaluate_batch(
        self,
        boards: Iterable,
        device: str | torch.device | None = None,
        histories: list[list] | None = None,
    ) -> list[tuple[dict, float, float]]:
        boards = list(boards)
        if not boards:
            return []
        histories = histories or [None] * len(boards)
        if len(histories) != len(boards):
            raise ValueError("histories must have the same length as boards")
        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device)
        was_training = self.training
        self.eval()
        try:
            legal_moves = [board.legal_moves() for board in boards]
            x = encode_boards(boards, histories=histories, device="cpu")
            mask = torch.stack([
                encode_move_mask(legal, board, device="cpu")
                for board, legal in zip(boards, legal_moves, strict=True)
            ])
            if device.type == "cuda":
                x = x.pin_memory().to(device, non_blocking=True)
                mask = mask.pin_memory().to(device, non_blocking=True)
            else:
                x = x.to(device)
                mask = mask.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = self(x, mask, return_dict=True)

            policy = out["policy"]
            # Bulk host transfer: fetch the whole output policy tensor to CPU in
            # a single operation, then extract per-row move probabilities from
            # CPU memory. This avoids launching one CUDA kernel per batch row
            # inside the Python loop below.
            policy_cpu = policy.float().cpu()
            values = out["value"].squeeze(-1).float().cpu().tolist()
            results = []
            for row, board, legal, value in zip(policy_cpu, boards, legal_moves, values, strict=True):
                if not legal:
                    results.append(({}, float(value), 0.0))
                    continue
                indices = torch.tensor(
                    [move_to_policy_index(board, move) for move in legal], dtype=torch.long
                )
                probabilities = row.index_select(0, indices).tolist()
                results.append((
                    {move: float(prob) for move, prob in zip(legal, probabilities, strict=True)},
                    float(value),
                    0.0,
                ))
            return results
        finally:
            self.train(was_training)


def _config_from_payload(config: dict) -> ModelConfig:
    valid = set(ModelConfig.__dataclass_fields__)
    return ModelConfig(**{key: value for key, value in config.items() if key in valid})


def load_model(path: str | Path, device: str | torch.device = "cpu") -> ZeroNet:
    payload = torch.load(path, map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    config_dict = dict(payload.get("config", {})) if isinstance(payload, dict) else {}
    if "stem.0.weight" in state:
        config_dict["channels"] = state["stem.0.weight"].shape[0]
        config_dict["input_channels"] = state["stem.0.weight"].shape[1]
    indices = [
        int(key.split(".")[1]) for key in state
        if key.startswith("tower.") and len(key.split(".")) > 1 and key.split(".")[1].isdigit()
    ]
    if indices:
        config_dict["blocks"] = max(indices) + 1
    if "policy_head.2.weight" in state:
        config_dict["policy_channels"] = state["policy_head.2.weight"].shape[0]
    if "policy_head.4.weight" in state:
        config_dict["policy_size"] = state["policy_head.4.weight"].shape[0] * 64
    if "categorical_value_head.weight" in state:
        config_dict["num_value_bins"] = state["categorical_value_head.weight"].shape[0]
    if "tower.0.se.fc1.weight" in state and "se_reduction" not in config_dict:
        hidden_times_two, channels = state["tower.0.se.fc1.weight"].shape
        hidden = hidden_times_two // 2
        # When the SE bottleneck is clamped to 32 channels the original
        # reduction is not recoverable from weights, so retain the default.
        config_dict["se_reduction"] = channels // hidden if hidden > 32 and channels % hidden == 0 else 8
    model = ZeroNet(_config_from_payload(config_dict)).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def save_model(path: str | Path, model: ZeroNet, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "architecture": "zero_x_swiglu_seresnet_v3",
        "config": asdict(model.config),
        "model_hash": model_hash(model),
        "model": model.state_dict(),
        **extra,
    }
    with tmp_path.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp_path.replace(path)


@torch.inference_mode()
def model_hash(model: ZeroNet) -> str:
    """Return a stable SHA-256 hash of the model state for experiment provenance."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class _TorchScriptDeploymentWrapper(nn.Module):
    def __init__(self, model: ZeroNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor):
        features = self.model.tower(self.model.stem(x))
        policy_logits = self.model.policy_head(features).masked_fill(legal_mask <= 0, -1e4)
        value_features = self.model.value_body(self.model.value_pool(features))
        value_logits = self.model.categorical_value_head(value_features)
        value_probs = torch.softmax(value_logits, dim=-1)
        scalar_value = (value_probs * self.model.bin_centers).sum(dim=-1, keepdim=True)
        wdl_logits = self.model.wdl_head(value_features)
        wdl = torch.softmax(wdl_logits, dim=-1)
        return policy_logits, scalar_value, wdl


def export_torchscript(path: str | Path, model: ZeroNet, device: str | torch.device = "cuda") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("ZERO-X CUDA deployment requires native bfloat16 support")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    deployment = _TorchScriptDeploymentWrapper(copy.deepcopy(model).eval()).to(device=dev, dtype=dtype)

    example_x = torch.zeros((1, model.config.input_channels, 8, 8), device=dev, dtype=dtype)
    example_mask = torch.ones((1, POLICY_SIZE), device=dev, dtype=torch.float32)
    tmp_path = path.with_suffix(".tmp")

    with torch.inference_mode():
        traced = torch.jit.trace(deployment, (example_x, example_mask), check_trace=False)
        traced.save(str(tmp_path))
    tmp_path.replace(path)
    return path
