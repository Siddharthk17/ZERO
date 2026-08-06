import pytest

torch = pytest.importorskip("torch")

from zero_chess import Board
from zero_chess.encoding import INPUT_CHANNELS, POLICY_SIZE, encode_board, encode_move_mask
from zero_chess.model import ConvResidualBlock, ModelConfig, ZeroNet, _TorchScriptDeploymentWrapper


def test_tiny_model_forward_shapes() -> None:
    board = Board.starting_position()
    x = encode_board(board).unsqueeze(0)
    model = ZeroNet(ModelConfig(channels=32, blocks=2))
    out = model(x)
    assert x.shape == (1, INPUT_CHANNELS, 8, 8)
    assert out["policy_logits"].shape == (1, POLICY_SIZE)
    assert out["value"].shape == (1, 1)
    assert out["wdl_logits"].shape == (1, 3)
    assert out["moves_left"].shape == (1, 1)
    assert 0.0 <= out["moves_left"].item() <= 1.0
    assert torch.allclose(out["value"], out["scalar_value"])


def test_masked_forward_returns_legal_policy_distribution() -> None:
    board = Board.starting_position()
    x = encode_board(board).unsqueeze(0)
    mask = encode_move_mask(board.legal_moves(), board).unsqueeze(0)
    model = ZeroNet(ModelConfig(channels=32, blocks=2))
    out = model(x, mask)
    policy, value, wdl = out["policy"], out["value"], out["wdl"]
    assert policy.shape == (1, POLICY_SIZE)
    assert value.shape == (1, 1)
    assert wdl.shape == (1, 3)
    assert torch.all(policy[mask == 0] == 0)
    assert torch.allclose(wdl.sum(dim=-1), torch.ones(1), atol=1e-5)
    assert torch.allclose(value, out["scalar_value"])


def test_deployment_wrapper_uses_the_training_value_definition() -> None:
    board = Board.starting_position()
    x = encode_board(board).unsqueeze(0)
    mask = encode_move_mask(board.legal_moves(), board).unsqueeze(0)
    model = ZeroNet(ModelConfig(channels=16, blocks=1, policy_channels=4)).eval()
    wrapper = _TorchScriptDeploymentWrapper(model).eval()
    with torch.no_grad():
        _policy, deployed_value, deployed_wdl = wrapper(x, mask)
        trained = model(x, mask)
    assert torch.allclose(deployed_wdl, trained["wdl"])
    assert torch.allclose(deployed_value, trained["value"])


def test_tower_block_pattern() -> None:
    model = ZeroNet(ModelConfig(channels=32, blocks=6))
    assert isinstance(model.tower[0], ConvResidualBlock)
    assert isinstance(model.tower[1], ConvResidualBlock)
    assert all(isinstance(block, ConvResidualBlock) for block in model.tower)
    assert model.parameter_count() > 0


def test_model_evaluate_batch_on_cuda_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = ZeroNet(ModelConfig(channels=32, blocks=2, policy_channels=8)).cuda()
    boards = [Board.starting_position(), Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")]
    out = model.evaluate_batch(boards, device="cuda")
    assert len(out) == 2
    for priors, value, uncertainty in out:
        assert priors
        assert -1.0 <= value <= 1.0
        assert uncertainty >= 0.0
