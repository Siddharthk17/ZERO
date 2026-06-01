"""High-performance, Non-Zero-Sum parallel batch PUCT Monte Carlo tree search."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from threading import Condition, Event, RLock, Thread
from typing import Protocol

from .board import Board
from .constants import WHITE, BLACK
from .move import Move
from .targets import DRAW_VALUE, apply_contempt, opponent_value

VIRTUAL_LOSS_VALUE = 3.0
VIRTUAL_LOSS_VISITS = 3

class Evaluator(Protocol):
    def evaluate_batch(self, boards: list[Board]) -> list[tuple[dict[Move, float], float, float]]:
        """Return ``(legal_priors, value, uncertainty)`` for each board."""

class UniformEvaluator:
    """Fallback evaluator for bootstrap self-play and CPU-only smoke tests."""

    def evaluate_batch(self, boards: list[Board]) -> list[tuple[dict[Move, float], float, float]]:
        out = []
        for board in boards:
            terminal = board.result_values()
            if terminal is not None:
                # Return the terminal value corresponding to the active player's turn
                val = terminal[0] if board.turn == WHITE else terminal[1]
                out.append(({}, val, 0.0))
                continue
            moves = board.legal_moves()
            prob = 1.0 / len(moves) if moves else 0.0
            out.append(({move: prob for move in moves}, 0.0, 1.0))
        return out

class NetworkEvaluator:
    """Synchronous model evaluator utilizing local thread locks."""

    def __init__(self, model, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self._lock = RLock()
        if hasattr(self.model, "eval"):
            self.model.eval()

    def evaluate_batch(self, boards: list[Board]) -> list[tuple[dict[Move, float], float, float]]:
        with self._lock:
            return self.model.evaluate_batch(boards, self.device)

@dataclass(slots=True)
class _EvalRequest:
    boards: list[Board]
    done: Event = field(default_factory=Event)
    results: list[tuple[dict[Move, float], float, float]] | None = None
    error: BaseException | None = None

class SharedBatchEvaluator:
    """Coalesces evaluator calls from multiple self-play threads into large GPU batches."""

    def __init__(
        self,
        model,
        device: str = "cuda",
        max_batch_size: int = 128,
        max_wait_ms: float = 2.0,
    ) -> None:
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000.0
        self._condition = Condition()
        self._queue: list[_EvalRequest] = []
        self._closed = False
        self.total_batches = 0
        self.total_positions = 0
        self.max_observed_batch = 0
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._worker = Thread(target=self._run, name="zero-gpu-batch-evaluator", daemon=True)
        self._worker.start()

    def evaluate_batch(self, boards: list[Board]) -> list[tuple[dict[Move, float], float, float]]:
        if not boards:
            return []
        request = _EvalRequest(list(boards))
        with self._condition:
            self._queue.append(request)
            self._condition.notify_all()
        request.done.wait()
        if request.error is not None:
            raise request.error
        return request.results or []

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def average_batch_size(self) -> float:
        if self.total_batches <= 0:
            return 0.0
        return self.total_positions / self.total_batches

    def _run(self) -> None:
        while True:
            requests = self._collect_requests()
            if requests is None:
                return
            self._evaluate_requests(requests)

    def _collect_requests(self) -> list[_EvalRequest] | None:
        with self._condition:
            while not self._queue and not self._closed:
                self._condition.wait()
            if self._closed and not self._queue:
                return None

            deadline = time.monotonic() + self.max_wait_seconds
            while self._queued_positions() < self.max_batch_size and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)

            requests: list[_EvalRequest] = []
            positions = 0
            while self._queue and (positions < self.max_batch_size or not requests):
                request = self._queue.pop(0)
                requests.append(request)
                positions += len(request.boards)
            return requests

    def _queued_positions(self) -> int:
        return sum(len(request.boards) for request in self._queue)

    def _evaluate_requests(self, requests: list[_EvalRequest]) -> None:
        flat_boards = [board for request in requests for board in request.boards]
        try:
            flat_results = self.model.evaluate_batch(flat_boards, self.device)
        except BaseException as exc:
            for request in requests:
                request.error = exc
                request.done.set()
            return

        self.total_batches += 1
        self.total_positions += len(flat_boards)
        self.max_observed_batch = max(self.max_observed_batch, len(flat_boards))

        offset = 0
        for request in requests:
            end = offset + len(request.boards)
            request.results = flat_results[offset:end]
            offset = end
            request.done.set()

@dataclass(slots=True)
class Node:
    prior_probability: float = 0.0
    visit_count: int = 0
    total_value: float = 0.0
    uncertainty: float = 0.0
    children: dict[Move, "Node"] = field(default_factory=dict)
    is_expanded: bool = False
    virtual_loss_count: int = 0

    @property
    def q(self) -> float:
        return 0.0 if self.visit_count <= 0 else self.total_value / self.visit_count

    @property
    def prior(self) -> float:
        return self.prior_probability

    @prior.setter
    def prior(self, value: float) -> None:
        self.prior_probability = value

    @property
    def value_sum(self) -> float:
        return self.total_value

    @value_sum.setter
    def value_sum(self, value: float) -> None:
        self.total_value = value

    @property
    def expanded(self) -> bool:
        return self.is_expanded

    def apply_virtual_loss(self) -> None:
        self.total_value -= VIRTUAL_LOSS_VALUE
        self.visit_count += VIRTUAL_LOSS_VISITS
        self.virtual_loss_count += 1

    def undo_virtual_loss(self) -> None:
        if self.virtual_loss_count <= 0:
            return
        self.total_value += VIRTUAL_LOSS_VALUE
        self.visit_count -= VIRTUAL_LOSS_VISITS
        self.virtual_loss_count -= 1

@dataclass(slots=True)
class SearchResult:
    move: Move | None
    visits: dict[Move, int]
    root: Node
    resigned: bool = False
    root_q_with_contempt: float = 0.0

    @property
    def policy(self) -> dict[Move, float]:
        total = sum(self.visits.values())
        if total <= 0:
            return {move: 0.0 for move in self.visits}
        return {move: count / total for move, count in self.visits.items()}

    def __iter__(self):
        yield self.move
        yield self.policy

@dataclass(slots=True)
class _Leaf:
    node: Node
    board: Board
    path: list[Node]

class MCTS:
    """Search orchestrator supporting tree-reuse and custom reward rules."""

    def __init__(
        self,
        network: Evaluator | object | None = None,
        c_puct: float = 1.5,
        batch_size: int = 16,
        add_noise: bool = True,
        resign_threshold: float = -0.95,
        simulations: int = 200,
        cpuct: float | None = None,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        use_transpositions: bool = False, # Defaulted to False to prevent cyclic graphs [2]
        rng: random.Random | None = None,
        **_: object,
    ) -> None:
        if cpuct is not None:
            c_puct = cpuct
        self.evaluator: Evaluator = self._coerce_evaluator(network)
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.add_noise_default = add_noise
        self.resign_threshold = resign_threshold
        self.simulations = simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.use_transpositions = use_transpositions
        self.rng = rng or random.Random()
        self.root = Node()
        self.root_hash: int | None = None
        self.transposition_table: dict[int, Node] = {}
        self._resign_streak: dict[int, int] = {}
        self.last_average_batch_size = 0.0
        self.last_batches = 0

    def _coerce_evaluator(self, network: object | None) -> Evaluator:
        if network is None:
            return UniformEvaluator()
        if hasattr(network, "evaluate_batch"):
            return network  # type: ignore[return-value]
        return NetworkEvaluator(network)

    def reset(self) -> None:
        """Iteratively tear down MCTS nodes to speed up GC and avoid recursion limit crashes."""
        def _clear_node(node: Node | None) -> None:
            if node is None:
                return
            visited = set()
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr is None:
                    continue
                curr_id = id(curr)
                if curr_id in visited:
                    continue
                visited.add(curr_id)
                if hasattr(curr, "children") and curr.children:
                    children = list(curr.children.values())
                    curr.children.clear()
                    for child in children:
                        if child is not None:
                            stack.append(child)
        if hasattr(self, "root") and self.root:
            _clear_node(self.root)
            self.root = Node()
        self.root_hash = None
        
        if hasattr(self, "transposition_table"):
            for node in list(self.transposition_table.values()):
                _clear_node(node)
            self.transposition_table.clear()
        
        if hasattr(self, "_resign_streak"):
            self._resign_streak.clear()

    def search(
        self,
        board: Board,
        num_simulations: int | None = None,
        temperature: float = 0.0,
        add_noise: bool | None = None,
        generation: int = 10,
        c_puct: float | None = None,
    ) -> SearchResult:
        budget = self.simulations if num_simulations is None else num_simulations
        self.c_puct = self.c_puct if c_puct is None else c_puct
        root = self._root_for(board)
        if not root.is_expanded and board.outcome() is None:
            self._expand_batch([_Leaf(root, board.copy(), [root])])
        if add_noise if add_noise is not None else self.add_noise_default:
            self._add_dirichlet_noise(root)

        simulations_done = 0
        batch_sizes: list[int] = []
        while simulations_done < budget:
            leaves, terminals = self._collect_batch(board, root, budget - simulations_done)
            if not leaves and not terminals:
                break
            for path, value in terminals:
                self._backpropagate(path, value)
            if leaves:
                self._expand_batch(leaves)
                batch_sizes.append(len(leaves))
            simulations_done += len(leaves) + len(terminals)

        self.last_batches = len(batch_sizes)
        self.last_average_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0
        visits = {move: child.visit_count for move, child in root.children.items()}
        resigned = self._should_resign(board, root, generation)
        move = None if resigned else self._select_move(visits, temperature)
        root_q_with_contempt = apply_contempt(root.q)
        return SearchResult(move, visits, root, resigned, root_q_with_contempt)

    def search_time(
        self,
        board: Board,
        milliseconds: int,
        temperature: float = 0.0,
        add_noise: bool = False,
        generation: int = 10,
    ) -> SearchResult:
        deadline = time.monotonic() + max(milliseconds, 1) / 1000.0
        root = self._root_for(board)
        if not root.is_expanded and board.outcome() is None:
            self._expand_batch([_Leaf(root, board.copy(), [root])])
        while time.monotonic() < deadline:
            self.search(board, self.batch_size, temperature, add_noise, generation)
        visits = {move: child.visit_count for move, child in root.children.items()}
        root_q_with_contempt = apply_contempt(root.q)
        return SearchResult(self._select_move(visits, temperature), visits, root, root_q_with_contempt=root_q_with_contempt)

    def _root_for(self, board: Board) -> Node:
        if self.root_hash is None and self.root.is_expanded:
            self.root_hash = board.zobrist_hash
            if self.use_transpositions:
                self.transposition_table[board.zobrist_hash] = self.root
            return self.root
        if self.root_hash == board.zobrist_hash:
            return self.root
        if self.use_transpositions and board.zobrist_hash in self.transposition_table:
            self.root = self.transposition_table[board.zobrist_hash]
        else:
            self.root = Node()
            if self.use_transpositions:
                self.transposition_table[board.zobrist_hash] = self.root
        self.root_hash = board.zobrist_hash
        return self.root

    def _collect_batch(
        self, board: Board, root: Node, remaining: int
    ) -> tuple[list[_Leaf], list[tuple[list[Node], float]]]:
        leaves: list[_Leaf] = []
        terminals: list[tuple[list[Node], float]] = []
        target = min(self.batch_size, remaining)
        for _ in range(target):
            sim_board = board.copy()
            node = root
            path = [node]
            while node.is_expanded and node.children:
                move, node = self._select_child(node)
                sim_board.push(move)
                path.append(node)
            terminal = sim_board.result_value(sim_board.turn)
            if terminal is not None:
                if terminal == DRAW_VALUE:
                    terminal = DRAW_VALUE
                terminals.append((path, terminal))
            else:
                for n in path:
                    n.apply_virtual_loss()
                leaves.append(_Leaf(node, sim_board, path))
        return leaves, terminals

    def _expand_batch(self, leaves: list[_Leaf]) -> None:
        if not leaves:
            return
        results = self.evaluator.evaluate_batch([leaf.board for leaf in leaves])
        for leaf, (priors, value, uncertainty) in zip(leaves, results, strict=True):
            for n in leaf.path:
                n.undo_virtual_loss()
            self._expand_with_priors(leaf.node, leaf.board, priors, uncertainty)
            self._backpropagate(leaf.path, value)

    def _expand_with_priors(
        self, node: Node, board: Board, priors: dict[Move, float], uncertainty: float
    ) -> None:
        node.uncertainty = uncertainty
        if node.is_expanded:
            return
        total_prior = sum(max(0.0, prior) for prior in priors.values())
        for move, prior in priors.items():
            normalized = max(0.0, prior) / total_prior if total_prior > 0 else 0.0
            child = None
            if self.use_transpositions:
                board.push(move)
                child = self.transposition_table.get(board.zobrist_hash)
                if child is None:
                    child = Node(prior_probability=normalized)
                    self.transposition_table[board.zobrist_hash] = child
                elif child.prior_probability == 0.0:
                    child.prior_probability = normalized
                board.pop()
            if child is None:
                child = Node(prior_probability=normalized)
            node.children[move] = child
        node.is_expanded = True

    def _select_child(self, node: Node) -> tuple[Move, Node]:
        c_puct = self.c_puct
        parent_visit = node.visit_count
        # Precompute square root once per parent selection step to reduce CPU overhead [2]
        sqrt_parent_visit = math.sqrt(max(1, parent_visit))
        
        best_score = -float("inf")
        best: tuple[Move, Node] | None = None
        
        for move, child in node.children.items():
            # Corrected: unvisited nodes default to -1.0 (standard draw) instead of opponent_value(0.0) (-30.0) [2]
            q_from_parent = opponent_value(child.total_value / child.visit_count) if child.visit_count > 0 else opponent_value(-1.0)
            
            exploration = c_puct * child.prior_probability * sqrt_parent_visit / (1 + child.visit_count)
            score = q_from_parent + exploration
            
            if score > best_score:
                best_score = score
                best = (move, child)
                
        if best is None:
            raise RuntimeError("cannot select from a node without children")
        return best

    def _puct_score(self, parent: Node, child: Node) -> float:
        parent_visit = parent.visit_count
        sqrt_parent_visit = math.sqrt(max(1, parent_visit))
        q_from_parent = opponent_value(child.total_value / child.visit_count) if child.visit_count > 0 else opponent_value(-1.0)
        exploration = self.c_puct * child.prior_probability * sqrt_parent_visit / (1 + child.visit_count)
        return q_from_parent + exploration

    def _backpropagate(self, path: list[Node], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += value
            value = opponent_value(value)

    def _select_move(self, visits: dict[Move, int], temperature: float) -> Move | None:
        if not visits:
            return None
        if temperature <= 1e-6:
            return max(visits.items(), key=lambda item: item[1])[0]
        moves = list(visits)
        weights = [max(visit, 0) ** (1.0 / temperature) for visit in visits.values()]
        if sum(weights) <= 0:
            weights = [1.0] * len(moves)
        return self.rng.choices(moves, weights=weights, k=1)[0]

    def _add_dirichlet_noise(self, root: Node) -> None:
        if not root.children:
            return
        noise = _dirichlet([self.dirichlet_alpha] * len(root.children), self.rng)
        for child, eta in zip(root.children.values(), noise, strict=True):
            child.prior_probability = (1 - self.dirichlet_epsilon) * child.prior_probability + self.dirichlet_epsilon * eta

    def _should_resign(self, board: Board, root: Node, generation: int) -> bool:
        if generation < 10 or self.resign_threshold <= -1.0:
            return False
        if root.q < self.resign_threshold:
            self._resign_streak[board.turn] = self._resign_streak.get(board.turn, 0) + 1
        else:
            self._resign_streak[board.turn] = 0
        return self._resign_streak.get(board.turn, 0) >= 10

    def advance_to(self, move: Move) -> None:
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root_hash = None
            self.transposition_table.clear()
            return
        self.root = Node()
        self.root_hash = None
        self.transposition_table.clear()

def _dirichlet(alphas: list[float], rng: random.Random) -> list[float]:
    samples = [rng.gammavariate(alpha, 1.0) for alpha in alphas]
    total = sum(samples)
    if total <= 0:
        return [1.0 / len(alphas)] * len(alphas)
    return [sample / total for sample in samples]