import pytest

torch = pytest.importorskip("torch")

from zero_chess import Board
from zero_chess.constants import parse_square
from zero_chess.encoding import INPUT_CHANNELS, POLICY_SIZE, encode_board, encode_boards, encode_move_mask, move_to_policy_index


def test_starting_position_encoding_shape_and_planes() -> None:
    board = Board()
    tensor = encode_board(board)
    assert tensor.shape == (119, 8, 8)
    assert tensor[0, 1].sum().item() == 8
    assert tensor[3, 0, 0].item() == 1
    assert tensor[3, 0, 7].item() == 1
    assert tensor[5, 0, 4].item() == 1
    assert tensor[6, 6].sum().item() == 8
    assert tensor[9, 7, 0].item() == 1
    assert tensor[9, 7, 7].item() == 1
    assert tensor[11, 7, 4].item() == 1
    assert tensor[112].min().item() == 1
    assert all(tensor[idx].min().item() == 1 for idx in range(113, 117))
    assert tensor[117].sum().item() == 0
    assert torch.allclose(tensor[118], torch.full((8, 8), 1 / 512))


def test_black_to_move_is_rotated_to_own_pieces() -> None:
    board = Board()
    board.push_uci("e2e4")
    tensor = encode_board(board)
    assert tensor[0, 1].sum().item() == 8
    assert tensor[5, 0, 3].item() == 1
    assert tensor[112].sum().item() == 0


def test_en_passant_and_castling_planes() -> None:
    board = Board.from_fen("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1")
    tensor = encode_board(board)
    assert tensor[117, :, 3].sum().item() == 8
    assert all(tensor[idx].min().item() == 1 for idx in range(113, 117))


def test_move_mask_marks_only_legal_moves() -> None:
    board = Board()
    legal = board.legal_moves()
    mask = encode_move_mask(legal, board)
    assert mask.shape == (POLICY_SIZE,)
    assert mask.sum().item() == 20
    for move in legal:
        assert mask[move_to_policy_index(board, move)].item() == 1
    assert mask[move_to_policy_index(board, next(m for m in legal if m.uci() == "e2e4"))].item() == 1


def test_black_perspective_castling_planes_not_swapped() -> None:
    # Black to move, only kingside rights for both sides (Kk): own_ks should be set, own_qs clear.
    board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R b Kk - 0 1")
    tensor = encode_board(board)
    extra = 112
    assert tensor[extra + 1].min().item() == 1.0  # own kingside (BK)
    assert tensor[extra + 2].sum().item() == 0.0  # own queenside (BQ) -- was swapped before fix
    assert tensor[extra + 3].min().item() == 1.0  # opponent kingside (WK)
    assert tensor[extra + 4].sum().item() == 0.0  # opponent queenside (WQ)


def test_black_perspective_queenside_only() -> None:
    # Black to move, only queenside rights (Qq): own_qs set, own_ks clear.
    board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R b Qq - 0 1")
    tensor = encode_board(board)
    extra = 112
    assert tensor[extra + 1].sum().item() == 0.0  # own kingside clear
    assert tensor[extra + 2].min().item() == 1.0  # own queenside set
    assert tensor[extra + 3].sum().item() == 0.0  # opponent kingside clear
    assert tensor[extra + 4].min().item() == 1.0  # opponent queenside set


def test_batch_encoding_matches_single() -> None:
    boards = [Board.starting_position(), Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")]
    batch = encode_boards(boards)
    assert batch.shape == (2, INPUT_CHANNELS, 8, 8)
    assert torch.allclose(batch[0], encode_board(boards[0]))
    assert torch.allclose(batch[1], encode_board(boards[1]))
