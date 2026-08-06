from zero_chess import Board
from zero_chess.replay import Experience, PrioritizedReplayBuffer


def make_exp(priority: float = 1.0) -> Experience:
    board = Board()
    policy = {move.uci(): 1 / len(board.legal_moves()) for move in board.legal_moves()}
    return Experience(board.fen(), policy, (0.0, 1.0, 0.0), priority=priority)


def test_replay_samples_with_importance_weights() -> None:
    replay = PrioritizedReplayBuffer(hot_capacity=16)
    for idx in range(8):
        replay.add(make_exp(priority=idx + 1))
    batch = replay.sample_with_weights(4)
    assert len(batch.experiences) == 4
    assert len(batch.indices) == 4
    assert all(0 < weight <= 1 for weight in batch.weights)


def test_replay_updates_priorities() -> None:
    replay = PrioritizedReplayBuffer(hot_capacity=16)
    replay.add(make_exp(priority=1))
    batch = replay.sample_with_weights(1)
    replay.update_priorities(batch.indices, [10.0])
    assert replay.hot[batch.indices[0]].priority > 1


def test_hot_overflow_stays_in_ram() -> None:
    replay = PrioritizedReplayBuffer(hot_capacity=2)
    replay.add(make_exp())
    replay.add(make_exp())
    replay.add(make_exp())
    assert replay.hot_size == 2
    assert len(replay) == 2
    batch = replay.sample_with_weights(4)
    assert batch.experiences


def test_sampling_is_detinistic_with_seed() -> None:
    import random
    replay = PrioritizedReplayBuffer(hot_capacity=32, rng=random.Random(12345))
    for _ in range(32):
        replay.add(make_exp())
    a = [e.fen for e in replay.sample(8)]
    replay2 = PrioritizedReplayBuffer(hot_capacity=32, rng=random.Random(12345))
    for _ in range(32):
        replay2.add(make_exp())
    b = [e.fen for e in replay2.sample(8)]
    assert a == b


def test_sum_tree_supports_non_power_of_two_capacity() -> None:
    from zero_chess.replay import SumTree

    tree = SumTree(3)
    tree.update(0, 1.0)
    tree.update(1, 2.0)
    tree.update(2, 3.0)
    assert tree.get(0.1) == 0
    assert tree.get(1.1) == 1
    assert tree.get(3.1) == 2


def test_stale_priority_update_cannot_touch_replaced_slot() -> None:
    replay = PrioritizedReplayBuffer(hot_capacity=1)
    replay.add(make_exp())
    sample = replay.sample_with_weights(1)
    replay.add(make_exp())
    replay.update_priorities(sample.indices, [100.0], sample.generations)
    assert replay.hot[0].priority < 2.0


def test_load_truncates_to_smaller_capacity(tmp_path) -> None:
    replay = PrioritizedReplayBuffer(hot_capacity=4)
    for _ in range(4):
        replay.add(make_exp())
    path = tmp_path / "replay.pkl"
    replay.save(path)
    loaded = PrioritizedReplayBuffer.load(path, hot_capacity=2)
    assert len(loaded) == 2


def test_replay_save_load_preserves_sampling_rng(tmp_path) -> None:
    replay = PrioritizedReplayBuffer(hot_capacity=8)
    for _ in range(8):
        replay.add(make_exp())
    replay.sample(3)
    path = tmp_path / "replay_rng.pkl"
    replay.save(path)
    loaded = PrioritizedReplayBuffer.load(path)
    assert replay.rng.getstate() == loaded.rng.getstate()
