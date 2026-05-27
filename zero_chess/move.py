"""Compact move representation."""

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
    from_sq: int
    to_sq: int
    promotion: str | None = None
    flags: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.from_sq < 64 or not 0 <= self.to_sq < 64:
            raise ValueError(f"move squares out of range: {self.from_sq}->{self.to_sq}")
        if self.promotion is not None and self.promotion.upper() not in {"N", "B", "R", "Q"}:
            raise ValueError(f"invalid promotion: {self.promotion!r}")
        if self.promotion is not None and self.promotion != self.promotion.upper():
            object.__setattr__(self, "promotion", self.promotion.upper())

    @property
    def is_capture(self) -> bool:
        return bool(self.flags & CAPTURE)

    @property
    def is_promotion(self) -> bool:
        return self.promotion is not None or bool(self.flags & PROMOTION)

    @property
    def is_en_passant(self) -> bool:
        return bool(self.flags & EN_PASSANT)

    @property
    def is_castling(self) -> bool:
        return bool(self.flags & (KING_CASTLE | QUEEN_CASTLE))

    def encode(self) -> int:
        promo = PROMO_TO_CODE[self.promotion]
        return self.from_sq | (self.to_sq << 6) | (promo << 12) | (self.flags << 16)

    @classmethod
    def decode(cls, value: int) -> "Move":
        from_sq = value & 0x3F
        to_sq = (value >> 6) & 0x3F
        promo = CODE_TO_PROMO[(value >> 12) & 0x7]
        flags = value >> 16
        return cls(from_sq, to_sq, promo, flags)

    def uci(self) -> str:
        suffix = self.promotion.lower() if self.promotion else ""
        return f"{square_name(self.from_sq)}{square_name(self.to_sq)}{suffix}"

    @classmethod
    def from_uci(cls, text: str) -> "Move":
        text = text.strip()
        if len(text) not in (4, 5):
            raise ValueError(f"invalid UCI move: {text!r}")
        promo = text[4].upper() if len(text) == 5 else None
        return cls(parse_square(text[:2]), parse_square(text[2:4]), promo)

    def __str__(self) -> str:
        return self.uci()
