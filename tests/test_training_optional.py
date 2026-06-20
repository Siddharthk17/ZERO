import pytest

torch = pytest.importorskip("torch")

from zero_chess import Board
from zero_chess.ema import EMATeacher
from zero_chess.ewc import ElasticWeightConsolidation
from zero_chess.model import ModelConfig, ZeroNet, load_model, save_model
from zero_chess.replay import Experience, PrioritizedReplayBuffer
from zero_chess.training import ContinuousLRScheduler, TrainConfig, make_optimizer, td_blended_target, train_step


def make_replay(size: int = 8) -> PrioritizedReplayBuffer:
    replay = PrioritizedReplayBuffer(hot_capacity=64)
    board = Board()
    policy = {move.uci(): 1 / len(board.legal_moves()) for move in board.legal_moves()}
    for _ in range(size):
        replay.add(Experience(board.fen(), policy, 0.5, 0.25, (1.0, 0.0, 0.0), priority=1.0))
    return replay


def test_td_blended_target_formula() -> None:
    assert td_blended_target(1.0, -0.5, 0.7) == pytest.approx(0.55)


def test_lr_schedule_values() -> None:
    model = ZeroNet(ModelConfig(channels=16, blocks=1, attention_heads=4))
    opt = make_optimizer(model, TrainConfig(batch_size=2, device="cpu"))
    scheduler = ContinuousLRScheduler(opt)
    assert scheduler.lr_at(0) == pytest.approx(1e-3)
    assert 3e-5 < scheduler.lr_at(250) < 1e-3
    assert scheduler.lr_at(500) == pytest.approx(3e-5)
    assert scheduler.lr_at(501) == pytest.approx(3e-5)


def test_training_step_runs_and_clips() -> None:
    model = ZeroNet(ModelConfig(channels=16, blocks=1, attention_heads=4))
    replay = make_replay()
    config = TrainConfig(batch_size=4, device="cpu", mixed_precision=False)
    opt = make_optimizer(model, config)
    ewc = ElasticWeightConsolidation()
    ewc.consolidate(model, replay, device="cpu")
    metrics, scaler = train_step(model, opt, replay, config, ewc=ewc, iteration=1)
    assert scaler is not None
    for key in ["policy_loss", "value_loss", "wdl_loss", "ewc_loss", "aux_loss", "loss"]:
        assert key in metrics
    assert metrics["grad_norm"] <= 1.0


def test_ema_and_checkpoint_round_trip(tmp_path) -> None:
    model = ZeroNet(ModelConfig(channels=16, blocks=1, attention_heads=4))
    teacher = EMATeacher(model)
    teacher.update(model)
    path = tmp_path / "model.pt"
    save_model(path, model)
    loaded = load_model(path)
    x = torch.zeros(1, 119, 8, 8)
    with torch.no_grad():
        a = model(x)["policy_logits"]
        b = loaded(x)["policy_logits"]
    assert torch.allclose(a, b)


def test_load_model_detects_transformer_every(tmp_path) -> None:
    for te in (3, 4, 5):
        cfg = ModelConfig(channels=16, blocks=8, attention_heads=4, transformer_every=te)
        model = ZeroNet(cfg)
        model.eval()
        path = tmp_path / f"model_te{te}.pt"
        save_model(path, model)
        loaded = load_model(path)
        assert loaded.config.transformer_every == te, f"te={te} not detected"
        assert loaded.config.blocks == 8
