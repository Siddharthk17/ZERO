"""Neural-network tensor and policy move encoding with high-speed indexing."""

from __future__ import annotations

from .board import Board
from .constants import BLACK, BK, BQ, WHITE, WK, WQ, color_of, file_of, piece_type, rank_of, square
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

PIECE_TO_PLANE = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
}

def orient_square(sq: int, turn: int) -> int:
    """Rotate the square 180 degrees for Black to align perspectives."""
    return sq if turn == WHITE else 63 - sq

def deorient_square(sq: int, turn: int) -> int:
    """Convert oriented coordinate back to actual board index."""
    return sq if turn == WHITE else 63 - sq

def move_to_policy_index(board: Board, move: Move) -> int:
    """Map a legal move to one of 4672 AlphaZero-style policy logits."""
    from_sq = orient_square(move.from_sq, board.turn)
    to_sq = orient_square(move.to_sq, board.turn)
    df = (to_sq & 7) - (from_sq & 7)
    dr = (to_sq >> 3) - (from_sq >> 3)

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
    """Map all legal moves in the position to their policy tensor indices."""
    return {move: move_to_policy_index(board, move) for move in board.legal_moves()}

def encode_board(board: Board, history: list[Board] | None = None, device: str | None = None):
    """Return a torch tensor of shape ``(119, 8, 8)`` oriented for the active player."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("encode_board requires PyTorch; install zero-chess[train]") from exc

    planes = torch.zeros((INPUT_CHANNELS, 8, 8), dtype=torch.float32, device=device)
    return encode_board_into(planes, board, history)

def encode_board_into(planes, board: Board, history: list[Board] | None = None):
    """Fill pre-allocated ``planes`` tensor with the board representation."""
    planes.zero_()
    positions = [board] + list(history or [])
    positions = positions[:HISTORY]
    perspective = board.turn

    for hist_idx, pos in enumerate(positions):
        base = hist_idx * 14
        for sq, piece in enumerate(pos.squares):
            if piece == ".":
                continue
            oriented = sq if perspective == WHITE else 63 - sq
            rank, file_ = oriented >> 3, oriented & 7
            
            plane_idx = PIECE_TO_PLANE[piece]
            if perspective == BLACK:
                plane_idx = (plane_idx + 6) % 12
                
            planes[base + plane_idx, rank, file_] = 1.0

        occurrences = pos.hash_history.count(pos.zobrist_hash)
        if occurrences >= 2:
            planes[base + 12].fill_(1.0)
        if occurrences >= 3:
            planes[base + 13].fill_(1.0)

    extra = HISTORY * 14
    if board.turn == WHITE:
        planes[extra].fill_(1.0)
        
    own_ks, own_qs, opp_ks, opp_qs = (WK, WQ, BK, BQ) if perspective == WHITE else (BQ, BK, WQ, WK)
    if board.castling_rights & own_ks:
        planes[extra + 1].fill_(1.0)
    if board.castling_rights & own_qs:
        planes[extra + 2].fill_(1.0)
    if board.castling_rights & opp_ks:
        planes[extra + 3].fill_(1.0)
    if board.castling_rights & opp_qs:
        planes[extra + 4].fill_(1.0)
        
    if board.ep_square is not None:
        ep = board.ep_square if perspective == WHITE else 63 - board.ep_square
        planes[extra + 5, :, ep & 7] = 1.0
    planes[extra + 6].fill_(min(board.fullmove_number, 512) / 512.0)
    return planes

def policy_target(board: Board, visits: dict[Move, int], device: str | None = None):
    """Generate the policy target plane from MCTS statistics."""
    try:
        import torch
    except ImportError as exc:
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
    """Generate a bool mask indicating legal moves in the policy output shape."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("policy_mask requires PyTorch; install zero-chess[train]") from exc

    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool, device=device)
    for move in board.legal_moves():
        mask[move_to_policy_index(board, move)] = True
    return mask

def encode_move_mask(legal_moves: list[Move] | None, board: Board, device: str | None = None):
    """Generate a float mask indicating legal moves in the policy output shape."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("encode_move_mask requires PyTorch; install zero-chess[train]") from exc

    mask = torch.zeros(POLICY_SIZE, dtype=torch.float32, device=device)
    return encode_move_mask_into(mask, legal_moves, board)

def encode_move_mask_into(mask, legal_moves: list[Move] | None, board: Board):
    """Fill pre-allocated float ``mask`` with legal move coordinates."""
    mask.zero_()
    for move in legal_moves if legal_moves is not None else board.legal_moves():
        mask[move_to_policy_index(board, move)] = 1.0
    return mask

def terminal_wdl(value: float) -> tuple[float, float, float]:
    """Map a value scalar into Win, Draw, Loss target probabilities."""
    from .targets import DRAW_VALUE

    if abs(value - DRAW_VALUE) < 1e-9:
        return (0.0, 1.0, 0.0)
    if value > DRAW_VALUE:
        return (1.0, 0.0, 0.0)
    return (0.0, 0.0, 1.0)