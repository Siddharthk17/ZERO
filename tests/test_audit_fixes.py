"""Regression tests for chess rules, checkpoints, targets, and edge cases."""

from __future__ import annotations

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
        "8/8/8/8/8/8/8/K6k w - - 0 1",  # insufficient material
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # starting
    ]
    for fen in fens:
        board = Board.from_fen(fen)
        legal = board.legal_moves()
        if not legal:
            assert board.outcome() is not None
        else:
            assert board.outcome() is None or board.outcome() == "1/2-1/2"


# CheckpointManager cleanup
def test_checkpoint_manager_cleanup_removes_stale_entries(tmp_path) -> None:
    from zero_chess.checkpoint import CheckpointManager
    from zero_chess.model import ModelConfig, ZeroNet

    mgr = CheckpointManager(tmp_path, keep_last=10, permanent_every=100)
    model = ZeroNet(ModelConfig(channels=16, blocks=1))

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
    model = ZeroNet(ModelConfig(channels=16, blocks=1))

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
    model = ZeroNet(ModelConfig(channels=16, blocks=1))
    mgr.save(model, iteration=10)
    mgr.save(model, iteration=20)

    # Delete the index file
    (tmp_path / "index.json").unlink()

    # Reconstruct should find the checkpoint files
    index = mgr._read_index()
    assert len(index) == 2
    iterations = sorted(item["iteration"] for item in index)
    assert iterations == [10, 20]


def test_checkpoint_manager_self_heals_valid_json_with_invalid_schema(tmp_path) -> None:
    from zero_chess.checkpoint import CheckpointManager
    from zero_chess.model import ModelConfig, ZeroNet

    mgr = CheckpointManager(tmp_path, keep_last=10, permanent_every=100)
    model = ZeroNet(ModelConfig(channels=16, blocks=1))
    mgr.save(model, iteration=10)
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    assert mgr.latest() is not None


# targets opponent_value edge cases
def test_opponent_value_at_all_anchors() -> None:
    from zero_chess.targets import opponent_value

    for x, expected_y in [(-1.0, 1.0), (-0.75, 0.75), (0.0, 0.0), (0.75, -0.75), (1.0, -1.0)]:
        result = opponent_value(x)
        assert abs(result - expected_y) < 1e-9, f"opponent_value({x}) = {result}, expected {expected_y}"


def test_opponent_value_continuous() -> None:
    """opponent_value is exact negation, hence continuous with no jumps."""
    from zero_chess.targets import opponent_value

    for x0, x1 in [(-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0)]:
        mid = (x0 + x1) / 2.0
        v0 = opponent_value(x0)
        v1 = opponent_value(x1)
        vm = opponent_value(mid)
        assert min(v0, v1) - 1e-9 <= vm <= max(v0, v1) + 1e-9, (
            f"Discontinuity at midpoint {mid}: v0={v0}, vm={vm}, v1={v1}"
        )


def test_opponent_value_is_exact_negation_outside_network_range() -> None:
    from zero_chess.targets import opponent_value

    assert opponent_value(-100.0) == 100.0
    assert opponent_value(100.0) == -100.0


def test_apply_contempt_boundaries() -> None:
    from zero_chess.targets import apply_contempt

    # Inside the search-only band: adds the bounded contempt bonus.
    assert apply_contempt(0.0) == pytest.approx(0.1)
    assert apply_contempt(0.1) == pytest.approx(0.2)
    assert apply_contempt(-0.1) == pytest.approx(0.0)
    # Outside the band: no change
    assert apply_contempt(0.16) == pytest.approx(0.16)
    assert apply_contempt(-0.16) == pytest.approx(-0.16)
    assert apply_contempt(1.0) == pytest.approx(1.0)
    assert apply_contempt(-1.0) == pytest.approx(-1.0)
    assert apply_contempt(2.0) == pytest.approx(1.0)


# encoding edge cases
def test_encode_boards_empty_list() -> None:
    pytest.importorskip("torch")
    from zero_chess.encoding import INPUT_CHANNELS, encode_boards

    batch = encode_boards([])
    assert batch.shape == (0, INPUT_CHANNELS, 8, 8)


def test_policy_target_empty_visits() -> None:
    pytest.importorskip("torch")
    from zero_chess.encoding import POLICY_SIZE, policy_target

    board = Board()
    target = policy_target(board, {})
    assert target.shape == (POLICY_SIZE,)
    assert abs(target.sum().item() - 1.0) < 1e-6


def test_policy_target_with_visits() -> None:
    pytest.importorskip("torch")
    from zero_chess.encoding import policy_target

    board = Board()
    legal = board.legal_moves()
    visits = {legal[0]: 8, legal[1]: 2}
    target = policy_target(board, visits)
    assert abs(target.sum().item() - 1.0) < 1e-6


# move edge cases
def test_move_from_uci_invalid_length() -> None:

    with pytest.raises(ValueError):
        Move.from_uci("e2e4e")
    with pytest.raises(ValueError):
        Move.from_uci("e2")


def test_move_lowercase_promotion_normalized() -> None:

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
        replay.add(Experience(board.fen(), policy, (1.0, 0.0, 0.0), priority=float(i + 1)))

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
