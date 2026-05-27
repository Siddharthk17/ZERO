from zero_chess import Board
from zero_chess.constants import parse_square


def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves():
        board.push(move)
        total += perft(board, depth - 1)
        board.pop()
    return total


def test_starting_position_perft() -> None:
    board = Board.starting_position()
    assert perft(board, 1) == 20
    assert perft(board, 2) == 400
    assert perft(board, 3) == 8902


def test_kiwipete_perft_depth_2() -> None:
    board = Board.from_fen("r3k2r/p1ppqpb1/bn2pnp1/2pPN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    assert perft(board, 1) == 48
    assert perft(board, 2) == 1991


def test_en_passant_capture_removes_pawn() -> None:
    board = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    move = next(move for move in board.legal_moves() if move.uci() == "e5d6")
    board.push(move)
    assert board.squares[parse_square("d5")] == "."
    assert board.squares[parse_square("d6")] == "P"


def test_castling_and_rights() -> None:
    board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert {"e1g1", "e1c1"} <= {move.uci() for move in board.legal_moves()}
    board.push_uci("e1g1")
    assert board.squares[parse_square("g1")] == "K"
    assert board.squares[parse_square("f1")] == "R"
    assert board.castling_fen() == "kq"


def test_underpromotions_are_legal() -> None:
    board = Board.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    assert {"a7a8q", "a7a8r", "a7a8b", "a7a8n"} <= {move.uci() for move in board.legal_moves()}


def test_mate_and_stalemate() -> None:
    mate = Board.from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert mate.outcome() == "1-0"
    stalemate = Board.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stalemate.outcome() == "1/2-1/2"


def test_insufficient_material_same_color_bishops() -> None:
    board = Board.from_fen("7k/8/8/8/8/4b3/8/K1B5 w - - 0 1")
    assert board.has_insufficient_material()


def test_threefold_repetition_hash_history() -> None:
    board = Board.from_fen("4k3/8/8/8/8/8/1n6/KN6 w - - 0 1")
    for uci in ["b1c3", "b2c4", "c3b1", "c4b2", "b1c3", "b2c4", "c3b1", "c4b2"]:
        board.push_uci(uci)
    assert board.is_threefold_repetition()


def test_board_default_is_starting_position() -> None:
    board = Board()
    assert len(board.legal_moves()) == 20
    assert board.fen() == Board.starting_position().fen()


def test_piece_move_generation_smoke() -> None:
    assert {"d4d5", "d4c5", "d4e5"} <= {m.uci() for m in Board.from_fen("4k3/8/8/2p1p3/3P4/8/8/4K3 w - - 0 1").legal_moves()}
    assert {"d4b5", "d4f5", "d4b3", "d4f3"} <= {m.uci() for m in Board.from_fen("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1").legal_moves()}
    assert "d4h8" in {m.uci() for m in Board.from_fen("7k/8/8/8/3B4/8/8/4K3 w - - 0 1").legal_moves()}
    assert "d4d8" in {m.uci() for m in Board.from_fen("3k4/8/8/8/3R4/8/8/4K3 w - - 0 1").legal_moves()}
    assert {"d4d8", "d4h8"} <= {m.uci() for m in Board.from_fen("3k3k/8/8/8/3Q4/8/8/4K3 w - - 0 1").legal_moves()}
    assert {"e1d1", "e1f1"} <= {m.uci() for m in Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").legal_moves()}


def test_all_castling_types_and_independent_rights() -> None:
    board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert {"e1g1", "e1c1"} <= {m.uci() for m in board.legal_moves()}
    board.push_uci("e1c1")
    assert board.castling_fen() == "kq"
    assert board.squares[parse_square("c1")] == "K"
    assert board.squares[parse_square("d1")] == "R"

    black = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    assert {"e8g8", "e8c8"} <= {m.uci() for m in black.legal_moves()}


def test_castling_conditions_violated() -> None:
    assert "e1g1" not in {m.uci() for m in Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w Qkq - 0 1").legal_moves()}
    assert "e1c1" not in {m.uci() for m in Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w Kkq - 0 1").legal_moves()}
    assert "e1g1" not in {m.uci() for m in Board.from_fen("r3k2r/8/8/8/8/8/8/R3KB1R w KQkq - 0 1").legal_moves()}
    assert "e1g1" not in {m.uci() for m in Board.from_fen("r3k2r/8/8/8/8/8/4r3/R3K2R w KQkq - 0 1").legal_moves()}
    assert "e1g1" not in {m.uci() for m in Board.from_fen("r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1").legal_moves()}
    assert "e1g1" not in {m.uci() for m in Board.from_fen("r3k2r/8/8/8/8/6r1/8/R3K2R w KQkq - 0 1").legal_moves()}


def test_en_passant_expires_after_one_move() -> None:
    board = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    board.push_uci("e1f1")
    assert board.ep_square is None


def test_en_passant_rank_pin_is_illegal() -> None:
    board = Board.from_fen("k7/8/8/K2pP2r/8/8/8/8 w - d6 0 1")
    assert "e5d6" not in {m.uci() for m in board.legal_moves()}


def test_double_check_allows_only_king_moves() -> None:
    board = Board.from_fen("4k3/8/8/1B6/8/8/8/4R2K b - - 0 1")
    assert board.is_check()
    assert board.legal_moves()
    assert all(move.from_sq == parse_square("e8") for move in board.legal_moves())


def test_absolute_pin_filters_off_ray_moves() -> None:
    board = Board.from_fen("4r3/8/8/8/8/8/4R3/4K3 w - - 0 1")
    moves = {m.uci() for m in board.legal_moves()}
    assert "e2d2" not in moves
    assert "e2e8" in moves


def test_scholars_mate_checkmate() -> None:
    board = Board()
    for uci in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]:
        board.push_uci(uci)
    assert board.outcome() == "1-0"
    assert board.is_checkmate()


def test_fifty_move_rule_and_resets() -> None:
    board = Board.from_fen("4k2r/8/8/8/8/8/8/R3K3 w Qk - 99 1")
    board.push_uci("a1a2")
    assert board.is_fifty_move_draw()

    pawn = Board.from_fen("4k3/8/8/8/8/8/P7/4K3 w - - 99 1")
    pawn.push_uci("a2a3")
    assert pawn.halfmove_clock == 0

    capture = Board.from_fen("r3k3/8/8/8/8/8/8/R3K3 w Q - 99 1")
    capture.push_uci("a1a8")
    assert capture.halfmove_clock == 0


def test_all_insufficient_material_draws_and_knight_pair_not_forced() -> None:
    assert Board.from_fen("8/8/8/8/8/8/8/K6k w - - 0 1").has_insufficient_material()
    assert Board.from_fen("8/8/8/8/8/8/8/KB5k w - - 0 1").has_insufficient_material()
    assert Board.from_fen("8/8/8/8/8/8/8/KN5k w - - 0 1").has_insufficient_material()
    assert Board.from_fen("7k/8/8/8/8/4b3/8/K1B5 w - - 0 1").has_insufficient_material()
    assert not Board.from_fen("7k/8/8/8/8/4n3/8/K1N5 w - - 0 1").has_insufficient_material()


def test_fen_round_trip_diverse_positions() -> None:
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "r3k2r/8/8/8/8/8/8/R3K2R w Qq - 10 20",
        "8/8/8/8/8/8/8/K6k b - - 99 50",
        "4rrk1/pp3ppp/2n5/3qp3/8/2NP1NP1/PP2PPBP/R2Q1RK1 w - - 4 12",
        "8/P7/8/8/8/8/7p/4K2k w - - 0 1",
        "2kr3r/ppp2ppp/2n5/8/8/8/PPP2PPP/R3K2R b KQ - 7 9",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        "4k3/8/8/8/8/8/1n6/KN6 w - - 12 34",
    ]
    for fen in fens:
        board = Board.from_fen(fen)
        assert Board.from_fen(board.fen()).fen() == fen
        assert Board(board.fen()).fen() == fen
        assert Board.from_fen(board.fen()).zobrist_hash == board.zobrist_hash


def test_zobrist_same_position_different_move_orders() -> None:
    a = Board()
    for uci in ["g1f3", "g8f6", "b1c3", "b8c6"]:
        a.push_uci(uci)
    b = Board()
    for uci in ["b1c3", "b8c6", "g1f3", "g8f6"]:
        b.push_uci(uci)
    assert a.fen() == b.fen()
    assert a.zobrist_hash == b.zobrist_hash


def test_zobrist_includes_castling_and_en_passant() -> None:
    with_rights = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    without_rights = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert with_rights.zobrist_hash != without_rights.zobrist_hash
    with_ep = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    without_ep = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1")
    assert with_ep.zobrist_hash != without_ep.zobrist_hash


def test_san_disambiguation_check_mate_and_castling() -> None:
    board = Board.from_fen("4k3/8/8/8/8/4K3/8/R6R w - - 0 1")
    san = {m.uci(): board.san(m) for m in board.legal_moves() if m.uci() in {"a1d1", "h1d1"}}
    assert san == {"a1d1": "Rad1", "h1d1": "Rhd1"}

    castle = Board.from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    assert castle.san(next(m for m in castle.legal_moves() if m.uci() == "e1g1")) == "O-O"
    assert castle.san(next(m for m in castle.legal_moves() if m.uci() == "e1c1")) == "O-O-O"

    check = Board.from_fen("6k1/8/6K1/8/8/8/5Q2/8 w - - 0 1")
    assert board_suffix(check, "f2f7") == "+"

    mate = Board()
    for uci in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6"]:
        mate.push_uci(uci)
    assert board_suffix(mate, "h5f7") == "#"


def board_suffix(board: Board, uci: str) -> str:
    move = next(m for m in board.legal_moves() if m.uci() == uci)
    san = board.san(move)
    return san[-1]
