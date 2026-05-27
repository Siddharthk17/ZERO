"""ZERO chess package.

High-performance, memory-optimized self-play reinforcement learning engine.
"""

from __future__ import annotations

from .board import Board
from .move import Move

try:
    from .model import ZERONetwork, ZeroNet
except ImportError:
    # PyTorch is optional if executing the pure-Python chess rules engine
    ZERONetwork = None  # type: ignore[assignment]
    ZeroNet = None  # type: ignore[assignment]

__all__ = ["Board", "Move", "ZERONetwork", "ZeroNet"]