"""Regression coverage for production pipeline hardening."""

from __future__ import annotations

import math
import random

import pytest

from zero_chess.board import Board
from zero_chess.replay import Experience, PrioritizedReplayBuffer, SumTree


def test_sum_tree_zero_boundary_selects_an_active_first_leaf() -> None:
    # Capacity need not be a power of two.  With the old <= comparison, an
    # exact zero sample could walk into an inactive leaf and be silently
    # coerced to the last live replay entry.
    tree = SumTree(3)
    tree.update(0, 1.0)
    assert tree.get(0.0) == 0


def test_experience_normalizes_wdl_and_rejects_non_finite_targets() -> None:
    board = Board()
    exp = Experience(board.fen(), {}, (2.0, 1.0, 1.0), q_mcts=4.0)
    assert exp.wdl == (0.5, 0.25, 0.25)
    assert exp.q_mcts == 1.0
    with pytest.raises(ValueError, match="finite"):
        Experience(board.fen(), {}, (1.0, math.nan, 0.0))


def test_experience_accepts_truncated_targets_without_relabeling_them() -> None:
    board = Board()
    exp = Experience(board.fen(), {}, (0.0, 1.0, 0.0), target_kind="truncated")
    assert exp.target_kind == "truncated"
    with pytest.raises(ValueError, match="target kind"):
        Experience(board.fen(), {}, (0.0, 1.0, 0.0), target_kind="unknown")


def test_model_evaluation_restores_training_mode() -> None:
    torch = pytest.importorskip("torch")
    from zero_chess.model import ModelConfig, ZeroNet

    model = ZeroNet(ModelConfig(channels=8, blocks=1, policy_channels=2))
    model.train()
    assert model.evaluate_batch([Board()], device="cpu")
    assert model.training
    assert torch.is_grad_enabled()


def test_uci_tracks_history_for_network_inputs() -> None:
    from zero_chess.uci import UCIEngine

    engine = UCIEngine()
    engine.handle("position startpos moves e2e4 e7e5 g1f3")
    assert len(engine.position_history) == 3
    assert engine.position_history[0].fen().startswith("rnbqkbnr/pppp1ppp/8/4p3")


def test_network_evaluator_receives_history_from_mcts() -> None:
    from zero_chess.mcts import MCTS, NetworkEvaluator

    class CaptureModel:
        def __init__(self) -> None:
            self.history_lengths: list[int] = []

        def eval(self) -> None:
            return None

        def evaluate_batch(self, boards, device, histories=None):
            self.history_lengths.extend(len(history or []) for history in (histories or []))
            return [
                ({move: 1.0 for move in board.legal_moves()}, 0.0, 0.0)
                for board in boards
            ]

    board = Board()
    previous = board.copy()
    board.push_uci("e2e4")
    model = CaptureModel()
    MCTS(NetworkEvaluator(model, "cpu"), add_noise=False).search(
        board,
        num_simulations=1,
        add_noise=False,
        history=[previous],
    )
    assert model.history_lengths
    assert max(model.history_lengths) >= 1


def test_arena_gate_reports_balanced_uniform_match() -> None:
    from zero_chess.arena import play_match
    from zero_chess.mcts import UniformEvaluator

    result = play_match(UniformEvaluator(), UniformEvaluator(), games=2, simulations=1, max_plies=8)
    assert result.games == 2
    assert result.wins_a + result.wins_b + result.draws == 2
    assert 0.0 <= result.score_low <= result.score_fraction <= result.score_high <= 1.0


def test_training_supports_non_default_value_bin_count() -> None:
    pytest.importorskip("torch")
    from zero_chess.model import ModelConfig, ZeroNet
    from zero_chess.training import TrainConfig, make_optimizer, train_step

    board = Board()
    policy = {move.uci(): 1.0 for move in board.legal_moves()}
    replay = PrioritizedReplayBuffer(hot_capacity=4, rng=random.Random(7))
    replay.add(Experience(board.fen(), policy, (0.0, 1.0, 0.0)))
    model = ZeroNet(ModelConfig(channels=8, blocks=1, policy_channels=2, num_value_bins=8))
    config = TrainConfig(batch_size=1, device="cpu", mixed_precision=False)
    metrics = train_step(model, make_optimizer(model, config), replay, config)
    assert math.isfinite(metrics["loss"])


def test_truncated_targets_train_value_without_wdl_label() -> None:
    pytest.importorskip("torch")
    from zero_chess.model import ModelConfig, ZeroNet
    from zero_chess.training import TrainConfig, make_optimizer, train_step

    board = Board()
    policy = {move.uci(): 1.0 for move in board.legal_moves()}
    replay = PrioritizedReplayBuffer(hot_capacity=2, rng=random.Random(8))
    replay.add(Experience(board.fen(), policy, (0.0, 1.0, 0.0), target_kind="truncated"))
    model = ZeroNet(ModelConfig(channels=8, blocks=1, policy_channels=2))
    config = TrainConfig(batch_size=1, device="cpu", mixed_precision=False)
    metrics = train_step(model, make_optimizer(model, config), replay, config)
    assert metrics["wdl_loss"] == 0.0
    assert metrics["terminal_target_fraction"] == 0.0


def test_two_hot_target_uses_input_dtype_and_validates_bin_count() -> None:
    torch = pytest.importorskip("torch")
    from zero_chess.training import compute_two_hot_target

    target = torch.tensor([0.0], dtype=torch.float64)
    encoded = compute_two_hot_target(target, num_bins=4)
    assert encoded.dtype == torch.float64
    assert torch.allclose(encoded.sum(dim=1), torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="at least 2"):
        compute_two_hot_target(target, num_bins=1)
