from zero_chess import Board
from zero_chess.mcts import MCTS, Node, UniformEvaluator


class CountingEvaluator:
    def __init__(self, value: float = 0.0) -> None:
        self.calls = 0
        self.batch_sizes = []
        self.value = value

    def evaluate_batch(self, boards):
        self.calls += 1
        self.batch_sizes.append(len(boards))
        out = []
        for board in boards:
            legal = board.legal_moves()
            priors = {move: (0.8 if idx == 0 else 0.2 / max(1, len(legal) - 1)) for idx, move in enumerate(legal)}
            out.append((priors, self.value, 0.0))
        return out


def test_puct_prefers_high_prior_unexplored_child() -> None:
    parent = Node(visit_count=10, is_expanded=True)
    low = Node(prior_probability=0.1)
    high = Node(prior_probability=0.9)
    mcts = MCTS(UniformEvaluator(), c_puct=1.5)
    assert mcts._puct_score(parent, high) > mcts._puct_score(parent, low)


def test_virtual_loss_apply_and_undo() -> None:
    node = Node(total_value=1.0, visit_count=2)
    node.apply_virtual_loss()
    assert node.total_value == -2.0
    assert node.visit_count == 5
    assert node.virtual_loss_count == 1
    node.undo_virtual_loss()
    assert node.total_value == 1.0
    assert node.visit_count == 2


def test_batch_mcts_uses_batched_evaluator_calls() -> None:
    evaluator = CountingEvaluator()
    result = MCTS(evaluator, batch_size=4, add_noise=False).search(Board(), num_simulations=8)
    assert result.move in Board().legal_moves()
    assert evaluator.batch_sizes[0] == 1
    assert max(evaluator.batch_sizes[1:]) == 4
    assert evaluator.calls <= 3


def test_dirichlet_noise_only_when_enabled() -> None:
    board = Board()
    no_noise = MCTS(UniformEvaluator(), add_noise=False, rng=__import__("random").Random(1))
    no_noise.search(board, num_simulations=0, add_noise=False)
    priors_a = [child.prior_probability for child in no_noise.root.children.values()]

    noise = MCTS(UniformEvaluator(), add_noise=True, rng=__import__("random").Random(1))
    noise.search(board, num_simulations=0, add_noise=True)
    priors_b = [child.prior_probability for child in noise.root.children.values()]
    assert priors_a != priors_b


def test_tree_reuse_preserves_child_stats() -> None:
    board = Board()
    mcts = MCTS(UniformEvaluator(), batch_size=2, add_noise=False)
    result = mcts.search(board, num_simulations=4)
    child = result.root.children[result.move]
    visits = child.visit_count
    board.push(result.move)
    mcts.advance_to(result.move)
    mcts.search(board, num_simulations=0)
    assert mcts.root is child
    assert mcts.root.visit_count == visits


def test_temperature_zero_and_nonzero_selection() -> None:
    board = Board()
    moves = board.legal_moves()[:2]
    mcts = MCTS(UniformEvaluator(), rng=__import__("random").Random(3))
    assert mcts._select_move({moves[0]: 1, moves[1]: 10}, 0.0) == moves[1]
    assert mcts._select_move({moves[0]: 1, moves[1]: 10}, 1.0) in moves


def test_resignation_after_ten_low_value_searches() -> None:
    mcts = MCTS(UniformEvaluator(), batch_size=2, resign_threshold=-0.5, add_noise=False)
    board = Board()
    mcts.root.total_value = -10
    mcts.root.visit_count = 10
    for _ in range(10):
        resigned = mcts._should_resign(board, mcts.root, generation=10)
    assert resigned


def test_collect_eval_requests_batch_constraints() -> None:
    import queue
    import threading
    import time
    import torch
    from zero_chess.self_play import _collect_eval_requests_nonblocking

    req_queue = queue.Queue()
    board = Board()
    tensor = torch.zeros(1, 119, 8, 8)
    mask = torch.zeros(1, 4672)

    # Worker 0 submits a request immediately
    req_queue.put(("eval", 0, 100, (tensor, mask)))

    collected = []
    def run_collector():
        res = _collect_eval_requests_nonblocking(req_queue, gpu_batch_size=32, wait_seconds=0.05)
        collected.append(res)

    t = threading.Thread(target=run_collector)
    t.start()

    # Wait 20ms, collector should still be waiting
    time.sleep(0.02)
    assert len(collected) == 0

    # Wait until after the deadline; collector should return the one request it has
    time.sleep(0.04)
    t.join(timeout=1.0)
    assert len(collected) == 1
    res = collected[0]
    assert len(res) == 1
    assert res[0][1] == 0


def test_reset_large_tree_does_not_overflow_recursion_limit() -> None:
    # Build a deep tree of nodes to exceed recursion limit (standard limit is 1000)
    # Let's create a chain of 2000 nodes
    mcts = MCTS(UniformEvaluator())
    curr = mcts.root
    from zero_chess.move import Move
    for i in range(2000):
        # We can mock a move
        dummy_move = Move.from_uci("e2e4")
        next_node = Node()
        curr.children[dummy_move] = next_node
        curr = next_node

    # This should clear successfully without RecursionError
    mcts.reset()
    assert len(mcts.root.children) == 0


def test_mcts_resign_threshold_is_respected() -> None:
    from zero_chess.self_play import SelfPlayConfig, _make_mcts

    config = SelfPlayConfig(disable_resign=False, resign_value=-0.5, simulations=1)
    mcts = _make_mcts(UniformEvaluator(), config)
    assert mcts.resign_threshold == -0.5

    config_disabled = SelfPlayConfig(disable_resign=True, resign_value=-0.5, simulations=1)
    mcts_disabled = _make_mcts(UniformEvaluator(), config_disabled)
    assert mcts_disabled.resign_threshold == -1.0

