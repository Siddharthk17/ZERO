import time

from zero_chess import Board
from zero_chess.mcts import MCTS, UniformEvaluator


def test_mcts_performance_log() -> None:
    start = time.perf_counter()
    mcts = MCTS(UniformEvaluator(), batch_size=16, add_noise=False)
    mcts.search(Board(), num_simulations=100, add_noise=False)
    elapsed = max(time.perf_counter() - start, 1e-9)
    print(
        {
            "simulations_per_second": 100 / elapsed,
            "gpu_utilization": "unavailable in unit test",
            "average_batch_size": mcts.last_average_batch_size,
        }
    )
