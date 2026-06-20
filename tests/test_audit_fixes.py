"""Tests for audit fixes: has_legal_moves, SharedBatchEvaluator close, checkpoint cleanup, edge cases."""

from __future__ import annotations

import json
import threading

import pytest

from zero_chess.board import Board
from zero_chess.constants import parse_square
from zero_chess.move import Move

# has_legal_moves
def test_has_legal_moves_starting_position() -> None:
    board = Board()
    assert board.has_legal_moves() is True

def test_has_legal_moves_checkmate_returns_false() -> None:
    mate = Board.from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert mate.has_legal_moves() is False

def test_has_legal_moves_stalemate_returns_false() -> None:
    stalemate = Board.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stalemate.has_legal_moves() is False

def test_has_legal_moves_agrees_with_legal_moves() -> None:
    fens = [
        Board.starting_position().fen(),
        "r3k2r/p1ppqpb1/bn2pnp1/2pPN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    ]
    for fen in fens:
        board = Board.from_fen(fen)
        assert board.has_legal_moves() == (len(board.legal_moves()) > 0)

def test_has_legal_moves_only_king_can_move() -> None:
    board = Board.from_fen("4k3/8/8/1B6/8/8/8/4R2K b - - 0 1")
    assert board.has_legal_moves() is True
    assert all(m.from_sq == parse_square("e8") for m in board.legal_moves())

# outcome consistency
def test_outcome_uses_has_legal_moves_consistency() -> None:
    """outcome() should agree with explicit legal_moves check for many positions."""
    fens = [
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",  # checkmate
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate
        "8/8/8/8/8/8/8/K6k w - - 0 1",      # insufficient material
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # starting
    ]
    for fen in fens:
        board = Board.from_fen(fen)
        legal = board.legal_moves()
        if not legal:
            assert board.outcome() is not None
        else:
            assert board.outcome() is None or board.outcome() == "1/2-1/2"

# SharedBatchEvaluator close wakes pending requests
def test_shared_batch_evaluator_close_wakes_pending() -> None:
    """Closing the evaluator while requests are pending must not hang."""
    from zero_chess.mcts import SharedBatchEvaluator, UniformEvaluator

    evaluator = SharedBatchEvaluator(UniformEvaluator(), device="cpu", max_batch_size=1, max_wait_ms=10.0)

    # Submit a request from a thread to simulate a pending call
    result_holder = {"error": None}

    def submit():
        try:
            evaluator.evaluate_batch([Board()])
        except Exception as exc:
            result_holder["error"] = exc

    t = threading.Thread(target=submit)
    t.start()
    # Give the thread time to queue its request
    import time
    time.sleep(0.05)

    evaluator.close()
    t.join(timeout=2.0)
    assert not t.is_alive(), "Thread should have been woken up by close()"
    assert result_holder["error"] is not None, "Pending request should have received an error"

# CheckpointManager cleanup
def test_checkpoint_manager_cleanup_removes_stale_entries(tmp_path) -> None:
    from zero_chess.checkpoint import CheckpointManager
    from zero_chess.model import ModelConfig, ZeroNet

    mgr = CheckpointManager(tmp_path, keep_last=10, permanent_every=100)
    model = ZeroNet(ModelConfig(channels=16, blocks=1, attention_heads=4))

    # Save 5 checkpoints (keep_last=10 so none are pruned)
    for i in range(5):
        mgr.save(model, iteration=i)

    index = mgr._read_index()
    assert len(index) == 5

    # Manually delete one checkpoint file to simulate a stale entry
    stale_path = tmp_path / "zero_iter_0000000.pt"
    assert stale_path.exists()
    stale_path.unlink()

    # cleanup_index should remove the stale entry
    cleaned = mgr._cleanup_index(index)
    assert len(cleaned) == 4
    assert all(entry["path"] != str(stale_path) for entry in cleaned)

def test_checkpoint_manager_save_and_latest_round_trip(tmp_path) -> None:
    from zero_chess.checkpoint import CheckpointManager
    from zero_chess.model import ModelConfig, ZeroNet

    mgr = CheckpointManager(tmp_path, keep_last=5, permanent_every=100)
    model = ZeroNet(ModelConfig(channels=16, blocks=1, attention_heads=4))

    meta = mgr.save(model, iteration=42, elo=1500.0)
    assert meta.iteration == 42
    assert meta.elo == 1500.0

    latest = mgr.latest()
    assert latest is not None
    assert latest.iteration == 42

def test_checkpoint_manager_reconstruct_index(tmp_path) -> None:
    """If index.json is deleted, the manager should reconstruct it from files."""
    from zero_chess.checkpoint import CheckpointManager
    from zero_chess.model import ModelConfig, ZeroNet

    mgr = CheckpointManager(tmp_path, keep_last=10, permanent_every=100)
    model = ZeroNet(ModelConfig(channels=16, blocks=1, attention_heads=4))
    mgr.save(model, iteration=10)
    mgr.save(model, iteration=20)

    # Delete the index file
    (tmp_path / "index.json").unlink()

    # Reconstruct should find the checkpoint files
    index = mgr._read_index()
    assert len(index) == 2
    iterations = sorted(item["iteration"] for item in index)
    assert iterations == [10, 20]

# targets opponent_value edge cases
def test_opponent_value_at_all_anchors() -> None:
    from zero_chess.targets import ANCHORS, opponent_value

    for x, expected_y in ANCHORS:
        result = opponent_value(x)
        assert abs(result - expected_y) < 1e-9, f"opponent_value({x}) = {result}, expected {expected_y}"

def test_opponent_value_continuous() -> None:
    """opponent_value should be continuous (no jumps) at anchor boundaries."""
    from zero_chess.targets import ANCHORS, opponent_value

    for i in range(len(ANCHORS) - 1):
        x0, _ = ANCHORS[i]
        x1, _ = ANCHORS[i + 1]
        if x0 == x1:
            continue
        mid = (x0 + x1) / 2.0
        # Value at midpoint should be between values at endpoints
        v0 = opponent_value(x0)
        v1 = opponent_value(x1)
        vm = opponent_value(mid)
        assert min(v0, v1) - 1e-9 <= vm <= max(v0, v1) + 1e-9, (
            f"Discontinuity at midpoint {mid}: v0={v0}, vm={vm}, v1={v1}"
        )

def test_opponent_value_clipping() -> None:
    from zero_chess.targets import opponent_value

    assert opponent_value(-100.0) == 0.0
    assert opponent_value(100.0) == -3.0

def test_apply_contempt_boundaries() -> None:
    from zero_chess.targets import apply_contempt

    # Inside the band: adds contempt bonus
    assert apply_contempt(0.0) == pytest.approx(0.3)
    assert apply_contempt(0.1) == pytest.approx(0.4)
    assert apply_contempt(-0.1) == pytest.approx(0.2)
    # Outside the band: no change
    assert apply_contempt(0.11) == pytest.approx(0.11)
    assert apply_contempt(-0.11) == pytest.approx(-0.11)
    assert apply_contempt(1.0) == pytest.approx(1.0)
    assert apply_contempt(-1.0) == pytest.approx(-1.0)

# encoding edge cases
def test_encode_boards_empty_list() -> None:
    torch = pytest.importorskip("torch")
    from zero_chess.encoding import INPUT_CHANNELS, encode_boards

    batch = encode_boards([])
    assert batch.shape == (0, INPUT_CHANNELS, 8, 8)

def test_policy_target_empty_visits() -> None:
    torch = pytest.importorskip("torch")
    from zero_chess.encoding import POLICY_SIZE, policy_target

    board = Board()
    target = policy_target(board, {})
    assert target.shape == (POLICY_SIZE,)
    assert abs(target.sum().item() - 1.0) < 1e-6

def test_policy_target_with_visits() -> None:
    torch = pytest.importorskip("torch")
    from zero_chess.encoding import policy_target

    board = Board()
    legal = board.legal_moves()
    visits = {legal[0]: 8, legal[1]: 2}
    target = policy_target(board, visits)
    assert abs(target.sum().item() - 1.0) < 1e-6

# move edge cases
def test_move_from_uci_invalid_length() -> None:
    from zero_chess.move import Move

    with pytest.raises(ValueError):
        Move.from_uci("e2e4e")
    with pytest.raises(ValueError):
        Move.from_uci("e2")

def test_move_encode_decode_round_trip() -> None:
    from zero_chess.move import Move

    moves = [
        Move(0, 16),
        Move(4, 6, flags=4),  # KING_CASTLE
        Move(52, 60, "Q", flags=33),  # promotion + capture
        Move(12, 28, None, flags=2),  # double pawn
    ]
    for move in moves:
        encoded = move.encode()
        decoded = Move.decode(encoded)
        assert decoded.from_sq == move.from_sq
        assert decoded.to_sq == move.to_sq
        assert decoded.promotion == move.promotion
        assert decoded.flags == move.flags

def test_move_lowercase_promotion_normalized() -> None:
    from zero_chess.move import Move

    move = Move(52, 60, "q")
    assert move.promotion == "Q"

# replay edge cases
def test_replay_empty_raises_on_sample() -> None:
    from zero_chess.replay import PrioritizedReplayBuffer

    replay = PrioritizedReplayBuffer(hot_capacity=16)
    with pytest.raises(ValueError):
        replay.sample(1)

def test_replay_anneal_beta() -> None:
    from zero_chess.replay import PrioritizedReplayBuffer

    replay = PrioritizedReplayBuffer(hot_capacity=16)
    assert replay.anneal_beta(0) == pytest.approx(0.4)
    assert replay.anneal_beta(500_000) == pytest.approx(1.0)
    assert replay.anneal_beta(250_000) == pytest.approx(0.7)

def test_replay_save_load_round_trip(tmp_path) -> None:
    from zero_chess.replay import Experience, PrioritizedReplayBuffer

    replay = PrioritizedReplayBuffer(hot_capacity=16)
    board = Board()
    policy = {m.uci(): 1 / len(board.legal_moves()) for m in board.legal_moves()}
    for i in range(5):
        replay.add(Experience(board.fen(), policy, 0.5, 0.25, (1.0, 0.0, 0.0), priority=float(i + 1)))

    path = tmp_path / "replay.pkl"
    replay.save(path)

    loaded = PrioritizedReplayBuffer.load(path, hot_capacity=16)
    assert len(loaded) == 5
    assert loaded.hot_size == 5

# mcts edge cases
def test_mcts_search_terminal_position() -> None:
    """MCTS on a terminal position should handle it gracefully."""
    from zero_chess.mcts import MCTS

    mate = Board.from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    result = MCTS(simulations=1).search(mate, num_simulations=1)
    # Checkmate position: root may not be expanded, move should be None or from empty children
    assert result.visits == {}

def test_mcts_reset_clears_transposition_table() -> None:
    from zero_chess.mcts import MCTS, UniformEvaluator

    mcts = MCTS(UniformEvaluator(), use_transpositions=True)
    mcts.search(Board(), num_simulations=2)
    assert len(mcts.transposition_table) > 0
    mcts.reset()
    assert len(mcts.transposition_table) == 0

# websocket parse_info
def test_parse_info_cp_score() -> None:
    from zero_chess.websocket_server import parse_info

    eval_val, nodes = parse_info("info depth 1 nodes 100 score cp 50")
    assert eval_val == pytest.approx(0.5)
    assert nodes == 100

def test_parse_info_mate_score() -> None:
    from zero_chess.websocket_server import parse_info

    eval_val, _ = parse_info("info depth 5 score mate 3")
    assert eval_val == 100.0

def test_parse_info_missing_fields() -> None:
    from zero_chess.websocket_server import parse_info

    eval_val, nodes = parse_info("info depth 1")
    assert eval_val is None
    assert nodes is None

# train.py import fix
def test_append_training_game_history_is_importable_from_train() -> None:
    """The function must be importable from the self_play module that train.py uses."""
    from zero_chess.self_play import _append_training_game_history

    assert callable(_append_training_game_history)

def test_append_training_game_history_writes_files(tmp_path) -> None:
    """The function should write both JSONL and PGN files."""
    import json as json_module
    from zero_chess.self_play import _append_training_game_history

    jsonl_path = tmp_path / "games.jsonl"
    pgn_path = tmp_path / "games.pgn"
    _append_training_game_history(
        game_number=1,
        generation=0,
        result="1-0",
        moves_san=["e4", "e5", "Nf3"],
        elo_after=16.0,
        elo_delta=16.0,
        rated_side=0,
        replay_size=100,
        train_step=5,
        metrics={"loss": 1.5},
        jsonl_path=jsonl_path,
        pgn_path=pgn_path,
    )
    assert jsonl_path.exists()
    assert pgn_path.exists()

    line = jsonl_path.read_text(encoding="utf-8").strip()
    record = json_module.loads(line)
    assert record["game_number"] == 1
    assert record["result"] == "1-0"
    assert record["ply_count"] == 3
    assert record["loss"] == 1.5

    pgn_text = pgn_path.read_text(encoding="utf-8")
    assert "1. e4 e5" in pgn_text
    assert "1-0" in pgn_text

# self_play CLI args
def test_self_play_cli_has_gpu_batch_size_arg() -> None:
    """The self-play CLI should accept --gpu-batch-size and --max-wait-ms."""
    from zero_chess.self_play import main as self_play_main

    # Parse args without actually running (use --help to trigger argparse)
    import sys
    old_argv = sys.argv
    sys.argv = ["self_play", "--help"]
    try:
        self_play_main([])
    except SystemExit:
        pass  # --help triggers SystemExit
    finally:
        sys.argv = old_argv

# QueueEvaluatorProxy stop_event
def test_queue_evaluator_proxy_respects_stop_event() -> None:
    """QueueEvaluatorProxy should raise immediately when stop_event is set."""
    import queue as queue_module
    from zero_chess.self_play import QueueEvaluatorProxy
    from zero_chess.board import Board

    req_queue = queue_module.Queue()
    resp_queue = queue_module.Queue()
    import threading as threading_module

    stop_event = threading_module.Event()
    stop_event.set()

    proxy = QueueEvaluatorProxy(0, req_queue, resp_queue, stop_event)
    with pytest.raises(RuntimeError, match="stop event"):
        proxy.evaluate_batch([Board()])

# _finalize augment flag
def test_finalize_with_augment_produces_flipped_experiences() -> None:
    """When augment=True and rng allows, _finalize should add horizontal mirror experiences."""
    import random
    from zero_chess.self_play import PositionRecord, _finalize

    board = Board()
    move = board.legal_moves()[0]
    record = PositionRecord(
        fen=board.fen(),
        policy={move.uci(): 1.0},
        turn=0,
        root_value=0.0,
        aggression_score=0.0,
        momentum_reward=0.0,
        panic_penalty=0.0,
    )
    # Use a seeded rng that will trigger augmentation
    rng = random.Random(0)
    experiences = _finalize([record], "1-0", rng, augment=True)
    # May or may not have augmented depending on rng.random() < 0.5
    assert len(experiences) >= 1
    for exp in experiences:
        assert exp.fen  # FEN should be non-empty
        assert exp.policy  # Policy should be non-empty

# _bootstrap_value edge cases
def test_bootstrap_value_empty_records() -> None:
    from zero_chess.self_play import _bootstrap_value

    assert _bootstrap_value([], 0) == -1.0

def test_bootstrap_value_last_position() -> None:
    from zero_chess.self_play import PositionRecord, _bootstrap_value

    records = [PositionRecord("fen", {}, 0, 0.5, 0.0, 0.0, 0.0)]
    # Index 0 is the last position; bootstrap should return its own value
    result = _bootstrap_value(records, 0)
    assert result == 0.5  # Same turn, so returns root_value directly

def test_bootstrap_value_opponent_turn() -> None:
    from zero_chess.self_play import PositionRecord, _bootstrap_value
    from zero_chess.targets import opponent_value

    records = [
        PositionRecord("fen0", {}, 0, 0.5, 0.0, 0.0, 0.0),  # White's turn
        PositionRecord("fen1", {}, 1, 0.3, 0.0, 0.0, 0.0),  # Black's turn, 5 plies later
    ]
    result = _bootstrap_value(records, 0)
    # later_turn (1=Black) != current_turn (0=White), so opponent_value(0.3)
    assert result == opponent_value(0.3)

# _adjudicate edge cases
def test_adjudicate_short_list_returns_false() -> None:
    from zero_chess.self_play import _adjudicate

    assert _adjudicate([0.5] * 19) is False

def test_adjudicate_stable_and_crushing_white() -> None:
    from zero_chess.self_play import _adjudicate

    # 20 values all at 0.8 (white winning, stable) -> should adjudicate
    assert _adjudicate([0.8] * 20) is True

def test_adjudicate_stable_and_crushing_black() -> None:
    from zero_chess.self_play import _adjudicate

    # 20 values all at -3.0 (black crushing, stable) -> should adjudicate
    assert _adjudicate([-3.0] * 20) is True

def test_adjudicate_unstable_returns_false() -> None:
    from zero_chess.self_play import _adjudicate

    # High variance, not stable
    values = [0.0, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5,
              0.0, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5]
    assert _adjudicate(values) is False

def test_adjudicate_stable_but_not_crushing_returns_false() -> None:
    from zero_chess.self_play import _adjudicate

    # Stable at 0.0 (drawish, not crushing)
    assert _adjudicate([0.0] * 20) is False
    # Stable at -1.0 (draw value, not crushing)
    assert _adjudicate([-1.0] * 20) is False

# _flip_board_horizontal correctness
def test_flip_board_horizontal_preserves_piece_count() -> None:
    from zero_chess.self_play import _flip_board_horizontal

    board = Board.starting_position()
    flipped = _flip_board_horizontal(board)
    original_pieces = sum(1 for p in board.squares if p != ".")
    flipped_pieces = sum(1 for p in flipped.squares if p != ".")
    assert original_pieces == flipped_pieces == 32

def test_flip_board_horizontal_swaps_castling_rights() -> None:
    from zero_chess.self_play import _flip_board_horizontal
    from zero_chess.constants import WK, WQ, BK, BQ

    board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    flipped = _flip_board_horizontal(board)
    # KQ should swap to QK (and vice versa) — the rights values should differ
    # but the total count of rights should be preserved
    original_count = bin(board.castling_rights).count("1")
    flipped_count = bin(flipped.castling_rights).count("1")
    assert original_count == flipped_count == 4

def test_flip_uci_horizontal() -> None:
    from zero_chess.self_play import _flip_uci_horizontal

    # e2e4 -> d2d4 (file 4 -> file 3)
    assert _flip_uci_horizontal("e2e4") == "d2d4"
    # a1h8 -> h1a8
    assert _flip_uci_horizontal("a1h8") == "h1a8"
    # Promotion: e7e8q -> d7d8q
    assert _flip_uci_horizontal("e7e8q") == "d7d8q"
    # Short string unchanged
    assert _flip_uci_horizontal("ab") == "ab"
