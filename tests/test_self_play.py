from zero_chess.model import ModelConfig, ZeroNet
from zero_chess.self_play import SelfPlayConfig, generate_multiprocess_games


def test_generate_multiprocess_games_runs_one_game() -> None:
    model = ZeroNet(ModelConfig(channels=32, blocks=2, attention_heads=2, policy_channels=8))
    config = SelfPlayConfig(
        simulations=2,
        batch_size=2,
        max_plies=20,
        temperature_moves=5,
        opening_random_plies=0,
    )
    games = generate_multiprocess_games(
        model,
        device="cpu",
        games=1,
        config=config,
        gpu_batch_size=2,
        max_wait_ms=5.0,
    )
    assert len(games) == 1
    result, experiences, sans, reason, meta = games[0]
    assert result in {"1-0", "0-1", "1/2-1/2"}
    assert len(experiences) > 0
    assert len(sans) > 0
    assert "duration" in meta
