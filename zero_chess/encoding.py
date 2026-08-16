"""Neural-network tensor and policy move encoding with high-speed indexing."""

from __future__ import annotations

from .board import Board
from .constants import BK, BLACK, BQ, WHITE, WK, WQ
from .move import Move

HISTORY = 8
INPUT_CHANNELS = HISTORY * 14 + 9
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

PIECE_TO_PLANE = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5, "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}


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
    """Return a torch tensor of shape ``(121, 8, 8)`` oriented for the active player."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("encode_board requires PyTorch; install the project runtime dependencies") from exc

    planes = torch.zeros((INPUT_CHANNELS, 8, 8), dtype=torch.float32, device=device)
    return encode_board_into(planes, board, history)


def encode_boards(boards: list[Board], histories: list[list[Board]] | None = None, device: str | None = None):
    """Batch-encode a list of boards into a tensor of shape ``(N, 121, 8, 8)``."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("encode_boards requires PyTorch; install the project runtime dependencies") from exc

    histories = histories or [None] * len(boards)
    batch = torch.zeros((len(boards), INPUT_CHANNELS, 8, 8), dtype=torch.float32, device=device)
    for idx, (board, history) in enumerate(zip(boards, histories, strict=True)):
        encode_board_into(batch[idx], board, history)
    return batch


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

    own_ks, own_qs, opp_ks, opp_qs = (WK, WQ, BK, BQ) if perspective == WHITE else (BK, BQ, WK, WQ)
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
    planes[extra + 7].fill_(min(board.halfmove_clock, 100) / 100.0)
    if board.is_check(perspective):
        planes[extra + 8].fill_(1.0)
    return planes


def policy_target(board: Board, visits: dict[Move, int], device: str | None = None):
    """Generate the policy target plane from MCTS statistics."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("policy_target requires PyTorch; install the project runtime dependencies") from exc

    target = torch.zeros(POLICY_SIZE, dtype=torch.float32, device=device)
    legal_by_move = {move: move_to_policy_index(board, move) for move in board.legal_moves()}
    valid_visits = {legal_by_move[move]: max(0, int(count)) for move, count in visits.items() if move in legal_by_move}
    total = sum(valid_visits.values())
    if total <= 0:
        legal = board.legal_moves()
        if not legal:
            return target
        prob = 1.0 / len(legal)
        for move in legal:
            target[move_to_policy_index(board, move)] = prob
        return target
    for index, count in valid_visits.items():
        target[index] = count / total
    return target


def encode_move_mask(legal_moves: list[Move] | None, board: Board, device: str | None = None):
    """Generate a float mask indicating legal moves in the policy output shape."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("encode_move_mask requires PyTorch; install the project runtime dependencies") from exc

    mask = torch.zeros(POLICY_SIZE, dtype=torch.float32, device=device)
    return encode_move_mask_into(mask, legal_moves, board)


def encode_move_mask_into(mask, legal_moves: list[Move] | None, board: Board):
    """Fill pre-allocated float ``mask`` with legal move coordinates."""
    mask.zero_()
    for move in legal_moves if legal_moves is not None else board.legal_moves():
        mask[move_to_policy_index(board, move)] = 1.0
    return mask


def terminal_wdl(value: float) -> tuple[float, float, float]:
    """Map a terminal minimax value to its one-hot WDL target."""
    from .targets import terminal_wdl_target

    return terminal_wdl_target(value)
