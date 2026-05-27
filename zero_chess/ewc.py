"""Elastic Weight Consolidation for continual learning."""

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
    def __init__(self, config: EWCConfig | None = None) -> None:
        self.config = config or EWCConfig()
        self.reference: dict[str, torch.Tensor] = {}
        self.fisher: dict[str, torch.Tensor] = {}

    def consolidate(self, model: torch.nn.Module, replay_buffer: PrioritizedReplayBuffer, device: str | torch.device | None = None) -> None:
        if len(replay_buffer) == 0:
            self.reference = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
            self.fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
            return
        device = device or next(model.parameters()).device
        model_was_training = model.training
        model.train()
        self.fisher = {name: torch.zeros_like(p, device=p.device) for name, p in model.named_parameters() if p.requires_grad}
        samples = replay_buffer.sample(min(self.config.sample_size, len(replay_buffer)))
        for start in range(0, len(samples), self.config.batch_size):
            chunk = samples[start : start + self.config.batch_size]
            x = torch.stack([encode_board(Board.from_fen(exp.fen), device=str(device)) for exp in chunk])
            targets = torch.stack([_policy_target(Board.from_fen(exp.fen), exp.policy, str(device)) for exp in chunk])
            model.zero_grad(set_to_none=True)
            out = model(x, return_dict=True)
            loss = -(targets * F.log_softmax(out["policy_logits"], dim=-1)).sum(dim=-1).mean()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and name in self.fisher:
                    self.fisher[name].add_(param.grad.detach().pow(2) * (len(chunk) / max(1, len(samples))))
        self.reference = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        model.train(model_was_training)

    def loss(self, model: torch.nn.Module) -> torch.Tensor:
        if not self.reference:
            return torch.zeros((), device=next(model.parameters()).device)
        total = torch.zeros((), device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.reference:
                total = total + (self.fisher[name] * (param - self.reference[name]).pow(2)).sum()
        return self.config.lambda_ * total


def _policy_target(board: Board, policy: dict[str, float], device: str) -> torch.Tensor:
    target = torch.zeros(POLICY_SIZE, dtype=torch.float32, device=device)
    total = sum(policy.values())
    if total <= 0:
        legal = board.legal_moves()
        if legal:
            for move in legal:
                target[move_to_policy_index(board, move)] = 1.0 / len(legal)
        return target
    for uci, prob in policy.items():
        target[move_to_policy_index(board, Move.from_uci(uci))] = float(prob) / total
    return target


EWC = ElasticWeightConsolidation
