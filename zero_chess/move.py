"""Compact, memory-optimized representation of a chess move."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import parse_square, square_name

CAPTURE = 1 << 0
DOUBLE_PAWN = 1 << 1
KING_CASTLE = 1 << 2
QUEEN_CASTLE = 1 << 3
EN_PASSANT = 1 << 4
PROMOTION = 1 << 5

PROMO_TO_CODE = {None: 0, "N": 1, "B": 2, "R": 3, "Q": 4}
CODE_TO_PROMO = {v: k for k, v in PROMO_TO_CODE.items()}


@dataclass(frozen=True, slots=True)
class Move:
    """Represents a lightweight, immutable, bit-packed chess move."""

    from_sq: int
    to_sq: int
    promotion: str | None = None
    flags: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.from_sq < 64 and 0 <= self.to_sq < 64):
            raise ValueError(f"move squares out of range: {self.from_sq}->{self.to_sq}")
            
        promo = self.promotion
        if promo is not None:
            # Fast-path check for pre-validated uppercase symbols (99% of internal moves)
            if promo not in ("N", "B", "R", "Q", "n", "b", "r", "q"):
                raise ValueError(f"invalid promotion piece symbol: {promo!r}")
            if not promo.isupper():
                object.__setattr__(self, "promotion", promo.upper())

    @property
    def is_capture(self) -> bool:
        """Check if the capture flag is set."""
        return bool(self.flags & CAPTURE)

    @property
    def is_promotion(self) -> bool:
        """Check if the move contains a promotion."""
        return self.promotion is not None or bool(self.flags & PROMOTION)

    @property
    def is_en_passant(self) -> bool:
        """Check if the en-passant flag is set."""
        return bool(self.flags & EN_PASSANT)

    @property
    def is_castling(self) -> bool:
        """Check if the move is castling."""
        return bool(self.flags & (KING_CASTLE | QUEEN_CASTLE))

    def encode(self) -> int:
        """Pack move attributes into a 32-bit integer."""
        promo = PROMO_TO_CODE[self.promotion]
        return self.from_sq | (self.to_sq << 6) | (promo << 12) | (self.flags << 16)

    @classmethod
    def decode(cls, value: int) -> "Move":
        """Unpack a 32-bit integer back into a structured Move object."""
        from_sq = value & 0x3F
        to_sq = (value >> 6) & 0x3F
        promo = CODE_TO_PROMO[(value >> 12) & 0x7]
        flags = value >> 16
        return cls(from_sq, to_sq, promo, flags)

    def uci(self) -> str:
        """Return the standard Universal Chess Interface (UCI) string (e.g. 'e2e4')."""
        suffix = self.promotion.lower() if self.promotion else ""
        return f"{square_name(self.from_sq)}{square_name(self.to_sq)}{suffix}"

    @classmethod
    def from_uci(cls, text: str) -> "Move":
        """Parse a standard Universal Chess Interface (UCI) string into a Move object."""
        text = text.strip()
        if len(text) not in (4, 5):
            raise ValueError(f"invalid UCI move format: {text!r}")
        promo = text[4].upper() if len(text) == 5 else None
        return cls(parse_square(text[:2]), parse_square(text[2:4]), promo)

    def __str__(self) -> str:
        return self.uci()