"""Deterministic Zobrist hashing tables."""

from __future__ import annotations

import random

PIECE_ORDER = "PNBRQKpnbrqk"
PIECE_INDEX = {piece: idx for idx, piece in enumerate(PIECE_ORDER)}

_rng = random.Random(0x5EED_5E1F_BADC_0DE)

PIECE_KEYS = [[_rng.getrandbits(64) for _ in range(64)] for _ in range(12)]
TURN_KEY = _rng.getrandbits(64)
CASTLING_KEYS = [_rng.getrandbits(64) for _ in range(16)]
EP_FILE_KEYS = [_rng.getrandbits(64) for _ in range(8)]


def mask64(value: int) -> int:
    return value & 0xFFFF_FFFF_FFFF_FFFF
