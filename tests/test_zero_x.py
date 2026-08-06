"""Regression tests for the ZERO-X zero-sum tabula-rasa pipeline."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zero_chess import Board
from zero_chess.mcts import MCTS, Node, UniformEvaluator
from zero_chess.model import ModelConfig, ZeroNet
from zero_chess.replay import PrioritizedReplayBuffer
from zero_chess.targets import (
    DRAW_VALUE,
    LOSS_VALUE,
    WIN_VALUE,
    game_result_to_values,
    opponent_value,
    terminal_wdl_target,
)
from zero_chess.training import ContinuousLRScheduler, TrainConfig, make_optimizer, train_step


def test_terminal_targets_are_exactly_zero_sum() -> None:
    assert game_result_to_values("1-0") == (WIN_VALUE, LOSS_VALUE)
    assert game_result_to_values("0-1") == (LOSS_VALUE, WIN_VALUE)
    assert game_result_to_values("1/2-1/2") == (DRAW_VALUE, DRAW_VALUE)
    assert opponent_value(0.73) == pytest.approx(-0.73)
    assert terminal_wdl_target(WIN_VALUE) == (1.0, 0.0, 0.0)
    assert terminal_wdl_target(DRAW_VALUE) == (0.0, 1.0, 0.0)
    assert terminal_wdl_target(LOSS_VALUE) == (0.0, 0.0, 1.0)


def test_backup_alternates_value_perspective_per_ply() -> None:
    root, child = Node(), Node()
    MCTS(UniformEvaluator())._backpropagate([root, child], 0.8, 1)
    assert child.q == pytest.approx(0.8)
    assert root.q == pytest.approx(-0.8)


def test_wdl_train_step_and_long_horizon_schedule() -> None:
    model = ZeroNet(ModelConfig(channels=8, blocks=1, policy_channels=2))
    replay = PrioritizedReplayBuffer(hot_capacity=16)
    board = Board()
    policy = {move.uci(): 1.0 for move in board.legal_moves()}
    from zero_chess.replay import Experience

    replay.extend([
        Experience(board.fen(), policy, terminal_wdl_target(DRAW_VALUE))
        for _ in range(4)
    ])
    config = TrainConfig(batch_size=4, device="cpu", mixed_precision=False)
    optimizer = make_optimizer(model, config)
    scheduler = ContinuousLRScheduler(
        optimizer, config.initial_lr, config.continuous_lr, config.total_steps, config.warmup_steps
    )
    assert scheduler.lr_at(0) == 0.0
    assert scheduler.lr_at(3_000) == pytest.approx(config.initial_lr)
    assert scheduler.lr_at(600_000) == pytest.approx(config.continuous_lr)
    metrics = train_step(model, optimizer, replay, config, iteration=3_000, scheduler=scheduler)
    assert metrics["wdl_loss"] > 0.0
    assert metrics["lr"] == pytest.approx(config.initial_lr)
