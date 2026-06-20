import pytest

torch = pytest.importorskip("torch")

from zero_chess import Board
from zero_chess.encoding import INPUT_CHANNELS, POLICY_SIZE, encode_board
from zero_chess.encoding import encode_move_mask
from zero_chess.model import ConvResidualBlock, ModelConfig, TransformerBlock, ZeroNet, ZERONetwork


def test_tiny_model_forward_shapes() -> None:
    board = Board.starting_position()
    x = encode_board(board).unsqueeze(0)
    model = ZeroNet(ModelConfig(channels=32, blocks=2, attention_heads=4))
    out = model(x)
    assert x.shape == (1, INPUT_CHANNELS, 8, 8)
    assert out["policy_logits"].shape == (1, POLICY_SIZE)
    assert out["value"].shape == (1, 1)
    assert out["wdl_logits"].shape == (1, 3)


def test_masked_forward_tuple_interface() -> None:
    board = Board.starting_position()
    x = encode_board(board).unsqueeze(0)
    mask = encode_move_mask(board.legal_moves(), board).unsqueeze(0)
    model = ZERONetwork(ModelConfig(channels=32, blocks=2, attention_heads=4))
    policy, value, wdl = model(x, mask)
    assert policy.shape == (1, POLICY_SIZE)
    assert value.shape == (1, 1)
    assert wdl.shape == (1, 3)
    assert torch.all(policy[mask == 0] == 0)
    assert torch.allclose(wdl.sum(dim=-1), torch.ones(1), atol=1e-5)


def test_tower_block_pattern() -> None:
    model = ZeroNet(ModelConfig(channels=32, blocks=6, attention_heads=4))
    assert isinstance(model.tower[0], ConvResidualBlock)
    assert isinstance(model.tower[1], ConvResidualBlock)
    assert isinstance(model.tower[2], TransformerBlock)
    assert isinstance(model.tower[5], TransformerBlock)
    assert model.parameter_count() > 0


def test_model_evaluate_batch_on_cuda_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = ZeroNet(ModelConfig(channels=32, blocks=2, attention_heads=2, policy_channels=8)).cuda()
    boards = [Board.starting_position(), Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")]
    out = model.evaluate_batch(boards, device="cuda")
    assert len(out) == 2
    for priors, value, uncertainty in out:
        assert priors
        assert -31.0 <= value <= 1.0
        assert uncertainty >= 0.0
