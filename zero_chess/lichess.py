"""Lichess deployment helpers."""

from __future__ import annotations


def default_uci_command(checkpoint: str = "checkpoints/latest.pt", device: str = "cuda", simulations: int = 800) -> str:
    return f"python -m zero_chess.uci --checkpoint {checkpoint} --device {device} --simulations {simulations}"
