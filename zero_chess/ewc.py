"""Elastic Weight Consolidation (EWC) for continual learning and forgetting prevention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .board import Board
from .encoding import POLICY_SIZE, encode_board, move_to_policy_index
from .move import Move
from .replay import PrioritizedReplayBuffer


@dataclass(slots=True)
class EWCConfig:
    lambda_: float = 0.1
    sample_size: int = 500
    batch_size: int = 32


class ElasticWeightConsolidation:
    """Computes and enforces quadratic penalties on weight drift from historical benchmarks."""

    def __init__(self, config: EWCConfig | None = None) -> None:
        self.config = config or EWCConfig()
        self.reference: dict[str, torch.Tensor] = {}
        self.fisher: dict[str, torch.Tensor] = {}

    def consolidate(
        self,
        model: torch.nn.Module,
        replay_buffer: PrioritizedReplayBuffer,
        device: str | torch.device | None = None,
    ) -> None:
        """Calculate the Fisher Information Matrix over a historical subset of the replay buffer."""
        if len(replay_buffer) == 0:
            self.reference = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
            self.fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
            return

        device = device or next(model.parameters()).device
        model_was_training = model.training
        model.train()

        self.fisher = {
            name: torch.zeros_like(p, device=p.device)
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        # Setup AMP parameters to keep GPU compute footprint extremely small
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        use_amp = device_type == "cuda"
        amp_dtype = torch.bfloat16 if device_type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

        samples = replay_buffer.sample(min(self.config.sample_size, len(replay_buffer)))
        total_samples = max(1, len(samples))

        for start in range(0, len(samples), self.config.batch_size):
            chunk = samples[start : start + self.config.batch_size]
            
            x = torch.stack([encode_board(Board.from_fen(exp.fen), device=str(device)) for exp in chunk])
            targets = torch.stack([_policy_target(Board.from_fen(exp.fen), exp.policy, str(device)) for exp in chunk])
            
            model.zero_grad(set_to_none=True)
            
            # Forward pass wrapped in mixed-precision context to leverage Tensor Cores safely
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                out = model(x, return_dict=True)
                loss = -(targets * F.log_softmax(out["policy_logits"], dim=-1)).sum(dim=-1).mean()
                
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None and name in self.fisher:
                    # Accumulate squared gradients representing parameters' importance
                    self.fisher[name].add_(
                        param.grad.detach().pow(2) * (len(chunk) / total_samples)
                    )

        self.reference = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        model.train(model_was_training)

    def loss(self, model: torch.nn.Module) -> torch.Tensor:
        """Compute the quadratic constraint loss penalizing deviations from benchmark parameters."""
        if not self.reference:
            return torch.zeros((), device=next(model.parameters()).device)
            
        total = torch.zeros((), device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.reference:
                total = total + (self.fisher[name] * (param - self.reference[name]).pow(2)).sum()
                
        return self.config.lambda_ * total


def _policy_target(board: Board, policy: dict[str, float], device: str) -> torch.Tensor:
    """Format and normalize the policy target vector for cross-entropy training."""
    target = torch.zeros(POLICY_SIZE, dtype=torch.float32, device=device)
    total = sum(policy.values())
    if total <= 0:
        legal = board.legal_moves()
        if legal:
            prob = 1.0 / len(legal)
            for move in legal:
                target[move_to_policy_index(board, move)] = prob
        return target
        
    for uci, prob in policy.items():
        target[move_to_policy_index(board, Move.from_uci(uci))] = float(prob) / total
    return target


EWC = ElasticWeightConsolidation