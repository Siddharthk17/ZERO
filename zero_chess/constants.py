"""Core constants shared by the chess engine."""

from __future__ import annotations

WHITE = 0
BLACK = 1

COLORS = (WHITE, BLACK)
COLOR_NAMES = {WHITE: "white", BLACK: "black"}

EMPTY = "."
PIECE_TYPES = "PNBRQK"
PROMOTION_TYPES = "QRBN"

WHITE_PIECES = set("PNBRQK")
BLACK_PIECES = set("pnbrqk")

WK = 1
WQ = 2
BK = 4
BQ = 8

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

KNIGHT_DELTAS = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

KING_DELTAS = (
    (1, 1),
    (1, 0),
    (1, -1),
    (0, 1),
    (0, -1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)

ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
QUEEN_DIRS = ROOK_DIRS + BISHOP_DIRS


def opposite(color: int) -> int:
    return BLACK if color == WHITE else WHITE


def color_of(piece: str) -> int | None:
    if piece in WHITE_PIECES:
        return WHITE
    if piece in BLACK_PIECES:
        return BLACK
    return None


def piece_type(piece: str) -> str:
    return piece.upper()


def piece_for(color: int, kind: str) -> str:
    return kind.upper() if color == WHITE else kind.lower()


def on_board(file_: int, rank: int) -> bool:
    return 0 <= file_ < 8 and 0 <= rank < 8


def square(file_: int, rank: int) -> int:
    return rank * 8 + file_


def file_of(sq: int) -> int:
    return sq & 7


def rank_of(sq: int) -> int:
    return sq >> 3


def square_name(sq: int) -> str:
    return f"{chr(ord('a') + file_of(sq))}{rank_of(sq) + 1}"


def parse_square(name: str) -> int:
    if len(name) != 2 or name[0] < "a" or name[0] > "h" or name[1] < "1" or name[1] > "8":
        raise ValueError(f"invalid square: {name!r}")
    return square(ord(name[0]) - ord("a"), int(name[1]) - 1)


def square_color(sq: int) -> int:
    return (file_of(sq) + rank_of(sq)) & 1
