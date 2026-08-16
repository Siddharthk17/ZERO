from zero_chess import Board
from zero_chess.encoding import move_to_policy_index
from zero_chess.replay import PrioritizedReplayBuffer
from zero_chess.rust_bridge import append_rust_game_history, ingest_rust_batch, sparse_policy_to_uci


def test_native_extension_board_smoke_when_built() -> None:
    import importlib

    import torch

    from zero_chess.encoding import encode_board, move_to_policy_index

    try:
        engine = importlib.import_module("zero_rust_engine")
    except ImportError:
        try:
            engine = importlib.import_module("zero_chess.zero_rust_engine")
        except ImportError:
            import pytest

            pytest.skip("native extension is not built")
    board = engine.FastRustBoard()
    assert len(board.legal_moves()) == 20
    python_board = Board.starting_position()
    native_encoding = torch.tensor(board.encode())
    python_encoding = encode_board(python_board).flatten()
    assert torch.equal(native_encoding, python_encoding)
    assert set(board.policy_indices()) == {
        move_to_policy_index(python_board, move) for move in python_board.legal_moves()
    }
    board.push_uci("e2e4")
    previous = python_board.copy()
    python_board.push_uci("e2e4")
    assert torch.equal(torch.tensor(board.encode()), encode_board(python_board, history=[previous]).flatten())

    castle = engine.FastRustBoard("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert "e1g1" in castle.legal_moves()
    castle.push_uci("e1g1")
    assert castle.fen().startswith("r3k2r/8/8/8/8/8/8/R4RK1 b")


def test_sparse_rust_policy_is_checked_against_legal_moves() -> None:
    board = Board.starting_position()
    move = next(move for move in board.legal_moves() if move.uci() == "e2e4")
    policy = sparse_policy_to_uci(board, [move_to_policy_index(board, move), 999_999], [0.75, 1.0])
    assert policy == {"e2e4": 1.0}


def test_ingest_rust_batch_builds_replay_experiences() -> None:
    board = Board.starting_position()
    move = next(move for move in board.legal_moves() if move.uci() == "e2e4")
    expected_index = move_to_policy_index(board, move)
    replay = PrioritizedReplayBuffer(hot_capacity=4)
    inserted = ingest_rust_batch(
        replay,
        {
            "games": [
                {
                    "experiences": [
                        {
                            "fen": board.fen(),
                            "policy_indices": [expected_index],
                            "policy_values": [1.0],
                            "value": 0.0,
                            "wdl": [0.0, 1.0, 0.0],
                            "material": [39.0, 0.0],
                            "moves_left": 0.42,
                            "opponent_policy_indices": [expected_index],
                            "opponent_policy_values": [1.0],
                            "history_fens": [],
                        }
                    ]
                }
            ]
        },
    )
    assert inserted == 1
    # Production policy targets use integer AlphaZero indices, not UCI strings,
    # so the Rust sparse format flows directly into the replay buffer.
    assert replay.hot[0].policy == {expected_index: 1.0}
    assert replay.hot[0].moves_left == 0.42
    assert replay.hot[0].material == (1.0, 0.0)
    assert replay.hot[0].opponent_policy == {expected_index: 1.0}


def test_ingest_rust_batch_clamps_promotion_excess_material() -> None:
    # Rust already normalises piece material by 39.0 in `piece_material`.
    # Positions with promoted pieces (e.g. 9 queens) legitimately exceed 1.0;
    # the bridge must saturate at unit scale via Experience clamping rather
    # than silently re-dividing by 39 a second time and corrupting the target.
    board = Board.starting_position()
    replay = PrioritizedReplayBuffer(hot_capacity=4)
    inserted = ingest_rust_batch(
        replay,
        {
            "games": [
                {
                    "experiences": [
                        {
                            "fen": board.fen(),
                            "policy_indices": [],
                            "policy_values": [],
                            "value": 0.0,
                            "wdl": [0.0, 1.0, 0.0],
                            "material": [1.5, 0.0],
                            "moves_left": 0.0,
                            "history_fens": [],
                        }
                    ]
                }
            ]
        },
    )
    assert inserted == 1
    assert replay.hot[0].material == (1.0, 0.0)


def test_ingest_rust_batch_preserves_truncated_target_kind() -> None:
    board = Board.starting_position()
    replay = PrioritizedReplayBuffer(hot_capacity=2)
    inserted = ingest_rust_batch(
        replay,
        {
            "games": [
                {
                    "experiences": [
                        {
                            "fen": board.fen(),
                            "policy_indices": [],
                            "policy_values": [],
                            "wdl": [0.0, 1.0, 0.0],
                            "target_kind": "truncated",
                        }
                    ]
                }
            ]
        },
    )
    assert inserted == 1
    assert replay.hot[0].target_kind == "truncated"


def test_native_game_history_records_target_provenance(tmp_path) -> None:
    path = tmp_path / "games.jsonl"
    count = append_rust_game_history(
        {
            "games": [
                {
                    "result": 0.0,
                    "termination": "max_plies",
                    "moves": ["e2e4"],
                    "experiences": [{"target_kind": "truncated"}],
                }
            ]
        },
        path,
        model_path="accepted.ts",
        batch_index=3,
    )
    assert count == 1
    assert '"target_counts":{"truncated":1}' in path.read_text(encoding="utf-8")
