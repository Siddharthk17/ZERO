"""Core constants and high-performance algebraic coordinate utilities."""

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
    """Return the opposite color using a fast bitwise XOR."""
    return color ^ 1

def color_of(piece: str) -> int | None:
    """Return the color of the piece, optimized to avoid set hashing overhead."""
    if piece == EMPTY:
        return None
    return WHITE if piece.isupper() else BLACK

def piece_type(piece: str) -> str:
    """Return the base uppercase representation of a piece."""
    return piece.upper()

def piece_for(color: int, kind: str) -> str:
    """Format piece symbol representation according to color."""
    return kind.upper() if color == WHITE else kind.lower()

def on_board(file_: int, rank: int) -> bool:
    """Verify if file and rank indices lie inside the 8x8 boundaries."""
    return 0 <= file_ < 8 and 0 <= rank < 8

def square(file_: int, rank: int) -> int:
    """Convert file and rank indices into a 0-63 square using fast bitwise shifts."""
    return (rank << 3) | file_

def file_of(sq: int) -> int:
    """Extract the file (column) index of a square [0-7]."""
    return sq & 7

def rank_of(sq: int) -> int:
    """Extract the rank (row) index of a square [0-7]."""
    return sq >> 3

def square_name(sq: int) -> str:
    """Return standard algebraic notation (e.g. 'e4') for a square index."""
    return f"{chr(97 + (sq & 7))}{(sq >> 3) + 1}"

def parse_square(name: str) -> int:
    """Convert standard algebraic notation (e.g. 'e4') into a 0-63 square index."""
    if len(name) != 2 or name[0] < "a" or name[0] > "h" or name[1] < "1" or name[1] > "8":
        raise ValueError(f"invalid square coordinate: {name!r}")
    return ((int(name[1]) - 1) << 3) | (ord(name[0]) - 97)

def square_color(sq: int) -> int:
    """Return 0 for dark squares, 1 for light squares."""
    return ((sq & 7) + (sq >> 3)) & 1