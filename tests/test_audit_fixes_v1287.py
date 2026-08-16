from zero_chess.constants import VIRTUAL_LOSS_VALUE, VIRTUAL_LOSS_VISITS
from zero_chess.mcts import MCTS, Node, UniformEvaluator
from zero_chess.model import ModelConfig, ZeroNet, load_model, save_model


def test_mcts_unvisited_default_parameterization() -> None:
    mcts = MCTS(UniformEvaluator())
    parent = Node(visit_count=10, is_expanded=True)
    unexplored_child = Node(prior_probability=0.5)

    # Compute PUCT score with different unvisited defaults
    score_default = mcts._puct_score(parent, unexplored_child, unvisited_default=-1.0)
    score_custom = mcts._puct_score(parent, unexplored_child, unvisited_default=0.0)

    # FPU is a value from the current parent's perspective, so the lower
    # unvisited value is selected less eagerly.
    assert score_default < score_custom


def test_squeeze_excitation_reduction_recovery(tmp_path) -> None:
    # Create config with low channel size to trigger max(channels // reduction, 8) clamping
    config = ModelConfig(channels=32, blocks=1, se_reduction=16)
    # Check that channels // se_reduction = 2, which clamps to 8 hidden channels
    model = ZeroNet(config)

    # Save the model
    model_path = tmp_path / "model_clamped.pt"
    save_model(model_path, model)

    # Load the model and check if the configuration's se_reduction recovers to 16
    loaded = load_model(model_path, device="cpu")
    assert loaded.config.se_reduction == 16


def test_centralization_of_magic_numbers() -> None:
    # Ensure VIRTUAL_LOSS_VALUE and VIRTUAL_LOSS_VISITS are imported correctly and match target values
    assert VIRTUAL_LOSS_VALUE == 3.0
    assert VIRTUAL_LOSS_VISITS == 3
