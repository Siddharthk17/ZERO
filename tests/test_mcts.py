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
    from zero_chess.self_play import _collect_eval_requests

    # Mock queue and active_workers
    req_queue = queue.Queue()
    active_workers = [True, True]

    # Worker 0 submits a request immediately
    board = Board()
    req_queue.put(("eval", 0, 100, [board]))

    collected = []
    def run_collector():
        res = _collect_eval_requests(req_queue, gpu_batch_size=32, wait_seconds=0.05, active_workers=active_workers)
        collected.append(res)

    t = threading.Thread(target=run_collector)
    t.start()

    # Wait 20ms, collector should still be waiting
    time.sleep(0.02)
    assert len(collected) == 0

    # Wait another 40ms (total 60ms, > 50ms deadline).
    # Since worker 1 hasn't submitted, collector should still be waiting.
    time.sleep(0.04)
    assert len(collected) == 0

    # Now worker 1 submits a request
    req_queue.put(("eval", 1, 101, [board]))

    # Now the collector should finish and return both requests
    t.join(timeout=1.0)
    assert len(collected) == 1
    res = collected[0]
    assert len(res) == 2
    assert res[0][1] == 0
    assert res[1][1] == 1


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

