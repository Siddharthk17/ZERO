"""Neural-network tensor and policy move encoding."""

from __future__ import annotations

from functools import lru_cache

from .board import Board
from .constants import BK, BQ, PIECE_TYPES, WHITE, WK, WQ, color_of, file_of, piece_type, rank_of, square
from .move import Move

HISTORY = 8
INPUT_CHANNELS = HISTORY * 14 + 7
POLICY_PLANES = 73
POLICY_SIZE = POLICY_PLANES * 64

QUEEN_DIRS_POLICY = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
KNIGHT_DIRS_POLICY = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
UNDERPROMOS = ("N", "B", "R")


def orient_square(sq: int, turn: int) -> int:
    return sq if turn == WHITE else 63 - sq


def deorient_square(sq: int, turn: int) -> int:
    return sq if turn == WHITE else 63 - sq


def move_to_policy_index(board: Board, move: Move) -> int:
    """Map a legal move to one of 4672 AlphaZero-style policy logits."""
    from_sq = orient_square(move.from_sq, board.turn)
    to_sq = orient_square(move.to_sq, board.turn)
    df = file_of(to_sq) - file_of(from_sq)
    dr = rank_of(to_sq) - rank_of(from_sq)

    if move.promotion in UNDERPROMOS:
        if dr != 1 or df not in (-1, 0, 1):
            raise ValueError(f"invalid underpromotion geometry: {move}")
        plane = 64 + UNDERPROMOS.index(move.promotion) * 3 + (df + 1)
        return plane * 64 + from_sq

    if (df, dr) in KNIGHT_DIRS_POLICY:
        plane = 56 + KNIGHT_DIRS_POLICY.index((df, dr))
        return plane * 64 + from_sq

    direction, distance = _queen_direction(df, dr)
    plane = QUEEN_DIRS_POLICY.index(direction) * 7 + (distance - 1)
    return plane * 64 + from_sq


def _queen_direction(df: int, dr: int) -> tuple[tuple[int, int], int]:
    if df == 0 and dr != 0:
        direction = (0, 1 if dr > 0 else -1)
        distance = abs(dr)
    elif dr == 0 and df != 0:
        direction = (1 if df > 0 else -1, 0)
        distance = abs(df)
    elif abs(df) == abs(dr) and df != 0:
        direction = (1 if df > 0 else -1, 1 if dr > 0 else -1)
        distance = abs(df)
    else:
        raise ValueError(f"move is neither queen-like nor knight-like: delta=({df},{dr})")
    if not 1 <= distance <= 7:
        raise ValueError(f"invalid queen-like distance: {distance}")
    return direction, distance


def legal_policy_indices(board: Board) -> dict[Move, int]:
    return {move: move_to_policy_index(board, move) for move in board.legal_moves()}


def encode_board(board: Board, history: list[Board] | None = None, device: str | None = None):
    """Return a torch tensor of shape ``(119, 8, 8)``.

    The current side to move is oriented upward. Piece planes are ordered as
    own P/N/B/R/Q/K, opponent P/N/B/R/Q/K, then two repetition planes for each
    history slot.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise RuntimeError("encode_board requires PyTorch; install zero-chess[train]") from exc

    planes = torch.zeros((INPUT_CHANNELS, 8, 8), dtype=torch.float32, device=device)
    return encode_board_into(planes, board, history)


def encode_board_into(planes, board: Board, history: list[Board] | None = None):
    """Fill ``planes`` with the board encoding and return it."""
    planes.zero_()
    positions = [board] + list(history or [])
    positions = positions[:HISTORY]
    perspective = board.turn

    for hist_idx, pos in enumerate(positions):
        base = hist_idx * 14
        for sq, piece in enumerate(pos.squares):
            if piece == ".":
                continue
            oriented = orient_square(sq, perspective)
            rank = rank_of(oriented)
            file_ = file_of(oriented)
            piece_color = color_of(piece)
            owner_offset = 0 if piece_color == perspective else 6
            type_offset = PIECE_TYPES.index(piece_type(piece))
            planes[base + owner_offset + type_offset, rank, file_] = 1.0

        occurrences = pos.hash_history.count(pos.zobrist_hash)
        if occurrences >= 2:
            planes[base + 12].fill_(1.0)
        if occurrences >= 3:
            planes[base + 13].fill_(1.0)

    extra = HISTORY * 14
    if board.turn == WHITE:
        planes[extra].fill_(1.0)
    if perspective == WHITE:
        own_ks, own_qs, opp_ks, opp_qs = WK, WQ, BK, BQ
    else:
        own_ks, own_qs, opp_ks, opp_qs = BQ, BK, WQ, WK
    if board.castling_rights & own_ks:
        planes[extra + 1].fill_(1.0)
    if board.castling_rights & own_qs:
        planes[extra + 2].fill_(1.0)
    if board.castling_rights & opp_ks:
        planes[extra + 3].fill_(1.0)
    if board.castling_rights & opp_qs:
        planes[extra + 4].fill_(1.0)
    if board.ep_square is not None:
        ep = orient_square(board.ep_square, perspective)
        planes[extra + 5, :, file_of(ep)] = 1.0
    planes[extra + 6].fill_(min(board.fullmove_number, 512) / 512.0)
    return planes


def policy_target(board: Board, visits: dict[Move, int], device: str | None = None):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("policy_target requires PyTorch; install zero-chess[train]") from exc

    target = torch.zeros(POLICY_SIZE, dtype=torch.float32, device=device)
    total = sum(visits.values())
    if total <= 0:
        legal = board.legal_moves()
        if not legal:
            return target
        prob = 1.0 / len(legal)
        for move in legal:
            target[move_to_policy_index(board, move)] = prob
        return target
    for move, count in visits.items():
        target[move_to_policy_index(board, move)] = count / total
    return target


def policy_mask(board: Board, device: str | None = None):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("policy_mask requires PyTorch; install zero-chess[train]") from exc

    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool, device=device)
    for move in board.legal_moves():
        mask[move_to_policy_index(board, move)] = True
    return mask


def encode_move_mask(legal_moves: list[Move] | None, board: Board, device: str | None = None):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("encode_move_mask requires PyTorch; install zero-chess[train]") from exc

    mask = torch.zeros(POLICY_SIZE, dtype=torch.float32, device=device)
    return encode_move_mask_into(mask, legal_moves, board)


def encode_move_mask_into(mask, legal_moves: list[Move] | None, board: Board):
    """Fill ``mask`` with legal move indicators and return it."""
    mask.zero_()
    for move in legal_moves if legal_moves is not None else board.legal_moves():
        mask[move_to_policy_index(board, move)] = 1.0
    return mask


@lru_cache(maxsize=1)
def horizontal_flip_policy_map() -> tuple[int, ...]:
    mapping = [0] * POLICY_SIZE
    for plane in range(POLICY_PLANES):
        for sq in range(64):
            mapping[plane * 64 + sq] = _flip_policy_index(plane, sq)
    return tuple(mapping)


def _flip_policy_index(plane: int, sq: int) -> int:
    flipped_sq = square(7 - file_of(sq), rank_of(sq))
    if plane < 56:
        dir_idx, dist_idx = divmod(plane, 7)
        df, dr = QUEEN_DIRS_POLICY[dir_idx]
        flipped_dir = (-df, dr)
        new_plane = QUEEN_DIRS_POLICY.index(flipped_dir) * 7 + dist_idx
    elif plane < 64:
        df, dr = KNIGHT_DIRS_POLICY[plane - 56]
        new_plane = 56 + KNIGHT_DIRS_POLICY.index((-df, dr))
    else:
        promo_idx, capture_idx = divmod(plane - 64, 3)
        new_capture_idx = 2 - capture_idx
        new_plane = 64 + promo_idx * 3 + new_capture_idx
    return new_plane * 64 + flipped_sq


def flip_policy_tensor(policy):
    mapping = horizontal_flip_policy_map()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("flip_policy_tensor requires PyTorch; install zero-chess[train]") from exc
    index = torch.tensor(mapping, dtype=torch.long, device=policy.device)
    return policy.index_select(-1, index)


def terminal_wdl(value: float) -> tuple[float, float, float]:
    from .targets import DRAW_VALUE

    if abs(value - DRAW_VALUE) < 1e-9:
        return (0.0, 1.0, 0.0)
    if value > DRAW_VALUE:
        return (1.0, 0.0, 0.0)
    return (0.0, 0.0, 1.0)
