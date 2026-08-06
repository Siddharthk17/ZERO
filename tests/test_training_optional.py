import pytest

torch = pytest.importorskip("torch")

from zero_chess import Board
from zero_chess.model import ModelConfig, ZeroNet, load_model, save_model
from zero_chess.replay import Experience, PrioritizedReplayBuffer
from zero_chess.training import ContinuousLRScheduler, TrainConfig, make_optimizer, train_step


def make_replay(size: int = 8) -> PrioritizedReplayBuffer:
    replay = PrioritizedReplayBuffer(hot_capacity=64)
    board = Board()
    policy = {move.uci(): 1 / len(board.legal_moves()) for move in board.legal_moves()}
    for _ in range(size):
        replay.add(Experience(board.fen(), policy, (1.0, 0.0, 0.0), priority=1.0))
    return replay


def test_lr_schedule_values() -> None:
    model = ZeroNet(ModelConfig(channels=16, blocks=1))
    opt = make_optimizer(model, TrainConfig(batch_size=2, device="cpu"))
    scheduler = ContinuousLRScheduler(opt)
    assert scheduler.lr_at(0) == pytest.approx(0.0)
    assert scheduler.lr_at(1_500) == pytest.approx(1e-3)
    assert scheduler.lr_at(3_000) == pytest.approx(2e-3)
    assert scheduler.lr_at(600_000) == pytest.approx(1e-5)


def test_training_step_runs_and_clips() -> None:
    model = ZeroNet(ModelConfig(channels=16, blocks=1))
    replay = make_replay()
    config = TrainConfig(batch_size=4, device="cpu", mixed_precision=False)
    opt = make_optimizer(model, config)
    metrics = train_step(model, opt, replay, config, iteration=1)
    for key in [
        "policy_loss", "value_loss", "wdl_loss", "moves_left_loss", "opponent_policy_loss", "value_error", "loss"
    ]:
        assert key in metrics
    assert metrics["grad_norm"] <= 1.0


def test_checkpoint_round_trip(tmp_path) -> None:
    model = ZeroNet(ModelConfig(channels=16, blocks=1))
    path = tmp_path / "model.pt"
    save_model(path, model)
    loaded = load_model(path)
    x = torch.zeros(1, 121, 8, 8)
    with torch.no_grad():
        a = model(x)["policy_logits"]
        b = loaded(x)["policy_logits"]
    assert torch.allclose(a, b)


def test_load_model_recovers_se_resnet_shape(tmp_path) -> None:
    model = ZeroNet(ModelConfig(channels=16, blocks=8, policy_channels=8))
    path = tmp_path / "model.pt"
    save_model(path, model)
    loaded = load_model(path)
    assert loaded.config.blocks == 8
    assert loaded.config.channels == 16
    assert loaded.config.policy_channels == 8
