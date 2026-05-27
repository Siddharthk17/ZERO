"""ZERO chess engine package."""

from .board import Board
from .move import Move

try:
    from .model import ZERONetwork, ZeroNet
except ImportError:  # PyTorch is optional for importing the pure chess core.
    ZERONetwork = None  # type: ignore[assignment]
    ZeroNet = None  # type: ignore[assignment]

__all__ = ["Board", "Move", "ZERONetwork", "ZeroNet"]
