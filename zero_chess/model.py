"""High-performance, memory-optimized hybrid Transformer-ResNet policy/value network."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .encoding import INPUT_CHANNELS, POLICY_SIZE


@dataclass(slots=True)
class ModelConfig:
    input_channels: int = INPUT_CHANNELS
    channels: int = 128  # Optimised for 150MB VRAM footprint on RTX 2050
    blocks: int = 6       # Blazing fast execution pipeline on i5-12450H
    transformer_every: int = 3
    attention_heads: int = 4
    policy_size: int = POLICY_SIZE
    dropout: float = 0.0
    se_reduction: int = 8       # Dynamic Squeeze & Excitation reduction
    policy_channels: int = 16   # Dynamic policy head intermediate channels


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for adaptive channel-wise feature recalibration."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        pooled = x.mean(dim=(2, 3))
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(pooled))))
        return x * scale.view(batch, channels, 1, 1)


class ConvResidualBlock(nn.Module):
    """Standard residual convolutional block with Batch Normalization and Squeeze-and-Excitation."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.se = SqueezeExcitation(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(F.silu(self.bn1(x)))
        y = self.conv2(F.silu(self.bn2(y)))
        return residual + self.se(y)


class BoardTransformerBlock(nn.Module):
    """Spatial Multi-Head Attention block mapping global dependencies across the 8x8 grid."""

    def __init__(self, channels: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, 64, channels))
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2) + self.pos
        y = self.norm1(tokens)
        attn, _ = self.attn(y, y, y, need_weights=False)
        tokens = tokens + attn
        tokens = tokens + self.ff(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class ZeroNet(nn.Module):
    """Dual-headed Transformer-ResNet orchestrator predicting legal policy and evaluation targets."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.input_channels, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        tower = []
        for idx in range(self.config.blocks):
            if (idx + 1) % self.config.transformer_every == 0:
                tower.append(BoardTransformerBlock(c, self.config.attention_heads, self.config.dropout))
            else:
                tower.append(ConvResidualBlock(c, reduction=self.config.se_reduction))
        self.tower = nn.Sequential(*tower)

        self.policy_head = nn.Sequential(
            nn.BatchNorm2d(c),
            nn.SiLU(),
            nn.Conv2d(c, self.config.policy_channels, kernel_size=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(self.config.policy_channels * 8 * 8, self.config.policy_size),
        )

        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c, c),
            nn.LayerNorm(c),
            nn.SiLU(),
            nn.Linear(c, c // 2),
            nn.LayerNorm(c // 2),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(c // 2, 1)
        self.wdl_head = nn.Linear(c // 2, 3)
        self.uncertainty_head = nn.Linear(c // 2, 1)
        self.material_head = nn.Linear(c // 2, 1)
        self.mobility_head = nn.Linear(c // 2, 1)
        self.king_safety_head = nn.Linear(c // 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        move_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ):
        y = self.tower(self.stem(x))
        policy_logits = self.policy_head(y)
        masked_logits = policy_logits
        if move_mask is not None:
            if move_mask.shape != policy_logits.shape:
                raise ValueError(f"move_mask shape {tuple(move_mask.shape)} != policy logits {tuple(policy_logits.shape)}")
            masked_logits = policy_logits.masked_fill(move_mask <= 0, -1e9)
        policy = torch.softmax(masked_logits, dim=-1)
        
        value_features = self.value_body(self.value_pool(y))
        
        # Maps directly to [-3.0, 1.0] matching the asymmetric draw-as-loss targets bounds
        value = 2.0 * torch.tanh(self.value_head(value_features)) - 1.0
        
        wdl_logits = self.wdl_head(value_features)
        wdl = torch.softmax(wdl_logits, dim=-1)
        output = {
            "policy_logits": policy_logits,
            "masked_policy_logits": masked_logits,
            "policy": policy,
            "value": value,
            "wdl_logits": wdl_logits,
            "wdl": wdl,
            "uncertainty": F.softplus(self.uncertainty_head(value_features)).squeeze(-1),
            "material": self.material_head(value_features).squeeze(-1),
            "mobility": self.mobility_head(value_features).squeeze(-1),
            "king_safety": self.king_safety_head(value_features).squeeze(-1),
        }
        if return_dict is None:
            return_dict = move_mask is None
        if return_dict:
            return output
        return policy, value, wdl

    def parameter_count(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.inference_mode()
    def evaluate_batch(self, boards, device: str | torch.device | None = None) -> list[tuple[dict, float, float]]:
        """Evaluate boards and return legal policy priors plus value/uncertainty."""
        from .encoding import encode_board, encode_move_mask, move_to_policy_index

        if device is None:
            device = next(self.parameters()).device
        self.eval()
        legal_moves = [board.legal_moves() for board in boards]
        if str(device).startswith("cuda"):
            tensors = torch.stack([encode_board(board, device="cpu") for board in boards]).pin_memory().to(device, non_blocking=True)
            masks = torch.stack(
                [encode_move_mask(legal, board, device="cpu") for board, legal in zip(boards, legal_moves, strict=True)]
            ).pin_memory().to(device, non_blocking=True)
        else:
            tensors = torch.stack([encode_board(board, device=str(device)) for board in boards])
            masks = torch.stack(
                [encode_move_mask(legal, board, device=str(device)) for board, legal in zip(boards, legal_moves, strict=True)]
            )
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        
        is_bf16 = False
        if device_type == "cuda":
            try:
                is_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            except Exception:
                pass
                
        amp_dtype = torch.bfloat16 if is_bf16 else torch.float16
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=device_type == "cuda"):
            out = self(tensors, masks, return_dict=True)
        policy = out["policy"]
        values = out["value"].squeeze(-1).detach().cpu().tolist()
        uncertainties = out["uncertainty"].detach().cpu().tolist()
        results = []
        for row, board, legal, value, uncertainty in zip(policy, boards, legal_moves, values, uncertainties, strict=True):
            if not legal:
                results.append(({}, float(value), float(uncertainty)))
                continue
            indices = torch.tensor([move_to_policy_index(board, move) for move in legal], device=row.device)
            probs = row.index_select(0, indices).detach().cpu().tolist()
            results.append(({move: float(prob) for move, prob in zip(legal, probs, strict=True)}, float(value), float(uncertainty)))
        return results


def load_model(path: str | Path, device: str | torch.device = "cpu") -> ZeroNet:
    """Load a model state payload from disk with automatic structural alignment."""
    payload = torch.load(path, map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    config_dict = payload.get("config", {}) if isinstance(payload, dict) else {}

    # Auto-detect architectural parameters from the loaded state_dict shapes
    if "stem.0.weight" in state:
        config_dict["channels"] = state["stem.0.weight"].shape[0]

    tower_indices = set()
    for key in state.keys():
        if key.startswith("tower."):
            parts = key.split(".")
            if parts[1].isdigit():
                tower_indices.add(int(parts[1]))
    if tower_indices:
        config_dict["blocks"] = max(tower_indices) + 1

    if "tower.0.se.fc1.weight" in state:
        hidden, channels = state["tower.0.se.fc1.weight"].shape
        config_dict["se_reduction"] = channels // hidden

    if "policy_head.2.weight" in state:
        config_dict["policy_channels"] = state["policy_head.2.weight"].shape[0]

    config = ModelConfig(**config_dict)
    model = ZeroNet(config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def save_model(path: str | Path, model: ZeroNet, **extra) -> None:
    """Save model weights atomically using temporary replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(model.config), "model": model.state_dict(), **extra}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def parameter_count(model: nn.Module) -> int:
    """Return parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


ResidualBlock = ConvResidualBlock
TransformerBlock = BoardTransformerBlock
ZERONetwork = ZeroNet