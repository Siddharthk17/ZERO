"""Parallel batched PUCT Monte Carlo tree search for a zero-sum value game."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from threading import Event
from typing import Protocol

from .board import Board
from .constants import VIRTUAL_LOSS_VALUE, VIRTUAL_LOSS_VISITS, WHITE
from .move import Move
from .targets import DRAW_VALUE, apply_contempt, opponent_value


class Evaluator(Protocol):
    """Protocol for batch position evaluators returning priors, value, and uncertainty."""

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
    """Synchronous model evaluator with lock-free forward pass.

    In eval mode, concurrent model invocations are safe because:
    - Parameters are read-only (no gradients)
    - CUDA default stream serializes kernel launches
    - Board encoding is stateless (no shared state)
    """

    def __init__(self, model, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        if hasattr(self.model, "eval"):
            self.model.eval()

    def evaluate_batch(
        self, boards: list[Board], histories: list[list[Board] | None] | None = None
    ) -> list[tuple[dict[Move, float], float, float]]:
        return self.model.evaluate_batch(boards, self.device, histories=histories)


@dataclass(slots=True)
class Node:
    """A node in the MCTS search tree tracking visits, values, priors, and children."""

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

    def apply_virtual_loss(self, value: float = -VIRTUAL_LOSS_VALUE) -> None:
        self.total_value += value
        self.visit_count += VIRTUAL_LOSS_VISITS
        self.virtual_loss_count += 1

    def undo_virtual_loss(self, value: float = -VIRTUAL_LOSS_VALUE) -> None:
        if self.virtual_loss_count <= 0:
            return
        self.total_value -= value
        self.visit_count -= VIRTUAL_LOSS_VISITS
        self.virtual_loss_count -= 1


@dataclass(slots=True)
class SearchResult:
    """Result of an MCTS search: selected move, visit counts, root node, and resignation flag."""

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
    history: list[Board]


class MCTS:
    """Search orchestrator with alternating minimax backups and FPU selection."""

    def __init__(
        self,
        network: Evaluator | object | None = None,
        c_puct: float = 1.25,
        batch_size: int = 32,
        add_noise: bool = True,
        resign_threshold: float = -0.95,
        simulations: int = 200,
        cpuct: float | None = None,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        fpu_reduction: float = 0.25,
        use_transpositions: bool = False,  # Defaulted to False to prevent cyclic graphs [2]
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
        self.fpu_reduction = fpu_reduction
        self.use_transpositions = use_transpositions
        self.rng = rng or random.Random()
        self.root = Node()
        self.root_board: Board | None = None
        self.root_hash: int | None = None
        self.root_context: tuple[object, ...] | None = None
        self.transposition_table: dict[tuple[object, ...], Node] = {}
        self._resign_streak: dict[int, int] = {}
        self._root_noise_applied = False
        self.last_average_batch_size = 0.0
        self.last_batches = 0

    def _coerce_evaluator(self, network: object | None) -> Evaluator:
        if network is None:
            return UniformEvaluator()
        if hasattr(network, "evaluate_batch") and not hasattr(network, "parameters"):
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
        self.root_board = None
        self.root_context = None
        self._root_noise_applied = False

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
        stop_event: Event | None = None,
        history: list[Board] | None = None,
    ) -> SearchResult:
        budget = self.simulations if num_simulations is None else num_simulations
        self.c_puct = self.c_puct if c_puct is None else c_puct
        history = history or []
        root = self._root_for(board, history)
        if not root.is_expanded and self._root_terminal(board, history) is None:
            self._expand_root(root, board, history)
        use_noise = add_noise if add_noise is not None else self.add_noise_default
        if use_noise and not self._root_noise_applied:
            self._add_dirichlet_noise(root)
            self._root_noise_applied = True

        simulations_done = 0
        batch_sizes: list[int] = []
        while simulations_done < budget and not (stop_event and stop_event.is_set()):
            leaves, terminals = self._collect_batch(
                board, root, budget - simulations_done, stop_event=stop_event, history=history
            )
            if not leaves and not terminals:
                break
            for path, value, leaf_turn in terminals:
                self._backpropagate(path, value, leaf_turn)
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
        history: list[Board] | None = None,
        stop_event: Event | None = None,
    ) -> SearchResult:
        deadline = time.monotonic() + max(milliseconds, 1) / 1000.0
        history = history or []
        root = self._root_for(board, history)
        if not root.is_expanded and self._root_terminal(board, history) is None:
            self._expand_root(root, board, history)
        last_result: SearchResult | None = None
        while time.monotonic() < deadline and not (stop_event and stop_event.is_set()):
            last_result = self.search(
                board,
                self.batch_size,
                temperature,
                add_noise,
                generation,
                history=history,
                stop_event=stop_event,
            )
            if last_result.resigned:
                return last_result
        visits = {move: child.visit_count for move, child in root.children.items()}
        root_q_with_contempt = apply_contempt(root.q)
        return SearchResult(
            self._select_move(visits, temperature),
            visits,
            root,
            resigned=last_result.resigned if last_result is not None else False,
            root_q_with_contempt=root_q_with_contempt,
        )

    def _root_terminal(self, board: Board, history: list[Board]) -> str | None:
        if _repetition_count(board, history, []) >= 3:
            return "1/2-1/2"
        return board.outcome()

    def _root_for(self, board: Board, history: list[Board] | None = None) -> Node:
        context = _board_context(board, history)
        if self.root_context is None:
            if self.root_board is not None and self.root_board.fen() != board.fen():
                self.root = Node()
            self.root_hash = board.zobrist_hash
            self.root_board = board.copy()
            self.root_context = context
            if self.use_transpositions:
                self.transposition_table[context] = self.root
            return self.root
        if self.root_context == context:
            return self.root
        if self.use_transpositions and context in self.transposition_table:
            self.root = self.transposition_table[context]
        else:
            self.root = Node()
            if self.use_transpositions:
                self.transposition_table[context] = self.root
        self.root_hash = board.zobrist_hash
        self.root_board = board.copy()
        self.root_context = context
        self._root_noise_applied = False
        return self.root

    def _collect_batch(
        self,
        board: Board,
        root: Node,
        remaining: int,
        stop_event: Event | None = None,
        history: list[Board] | None = None,
    ) -> tuple[list[_Leaf], list[tuple[list[Node], float, int]]]:
        leaves: list[_Leaf] = []
        terminals: list[tuple[list[Node], float, int]] = []
        target = min(self.batch_size, remaining)
        for _ in range(target):
            if stop_event and stop_event.is_set():
                break
            sim_board = board.copy()
            sim_history = [position.copy() for position in (history or [])[:7]]
            branch_positions: list[tuple[str, int, int, int | None]] = []
            node = root
            path = [node]
            is_repetition = False
            while node.is_expanded and node.children:
                move, node = self._select_child(node)
                previous = sim_board.copy()
                branch_positions.append(sim_board._position_key())
                sim_board._push_unchecked(move)
                sim_history.insert(0, previous)
                del sim_history[7:]
                path.append(node)
                if _repetition_count(sim_board, history or [], branch_positions) >= 3:
                    is_repetition = True
                    break
            if is_repetition:
                terminals.append((path, DRAW_VALUE, sim_board.turn))
            else:
                terminal = sim_board.result_value(sim_board.turn)
                if terminal is not None:
                    terminals.append((path, terminal, sim_board.turn))
                else:
                    for depth, n in enumerate(path):
                        n.apply_virtual_loss(-VIRTUAL_LOSS_VALUE if depth == 0 else VIRTUAL_LOSS_VALUE)
                    leaves.append(_Leaf(node, sim_board, path, sim_history))
        return leaves, terminals

    def _expand_batch(self, leaves: list[_Leaf]) -> None:
        if not leaves:
            return
        try:
            results = self._evaluate_leaves(leaves)
            if len(results) != len(leaves):
                raise RuntimeError("evaluator returned a result count different from its input batch")
            for leaf, (priors, value, uncertainty) in zip(leaves, results, strict=True):
                self._expand_with_priors(leaf.node, leaf.board, priors, uncertainty)
                self._backpropagate(leaf.path, value, leaf.board.turn)
        finally:
            for leaf in leaves:
                for depth, node in enumerate(leaf.path):
                    node.undo_virtual_loss(-VIRTUAL_LOSS_VALUE if depth == 0 else VIRTUAL_LOSS_VALUE)

    def _evaluate_leaves(self, leaves: list[_Leaf]):
        boards = [leaf.board for leaf in leaves]
        histories = [leaf.history for leaf in leaves]
        if isinstance(self.evaluator, NetworkEvaluator):
            return self.evaluator.evaluate_batch(boards, histories)
        return self.evaluator.evaluate_batch(boards)

    def _expand_root(self, root: Node, board: Board, history: list[Board]) -> None:
        """Expand a root for move priors without treating it as a simulation."""
        if isinstance(self.evaluator, NetworkEvaluator):
            result = self.evaluator.evaluate_batch([board], [history])[0]
        else:
            result = self.evaluator.evaluate_batch([board])[0]
        priors, _value, uncertainty = result
        self._expand_with_priors(root, board, priors, uncertainty)

    def _expand_with_priors(self, node: Node, board: Board, priors: dict[Move, float], uncertainty: float) -> None:
        node.uncertainty = uncertainty
        if node.is_expanded:
            return
        legal_moves = board.legal_moves()
        legal_priors = {
            move: float(priors.get(move, 0.0)) for move in legal_moves if math.isfinite(float(priors.get(move, 0.0)))
        }
        total_prior = sum(max(0.0, prior) for prior in legal_priors.values())
        for move in legal_moves:
            prior = legal_priors.get(move, 0.0)
            normalized = max(0.0, prior) / total_prior if total_prior > 0 else 1.0 / len(legal_moves)
            child = None
            if self.use_transpositions:
                board._push_unchecked(move)
                child_context = _board_context(board)
                child = self.transposition_table.get(child_context)
                if child is None:
                    child = Node(prior_probability=normalized)
                    self.transposition_table[child_context] = child
                elif child.prior_probability == 0.0:
                    child.prior_probability = normalized
                board.pop()
            if child is None:
                child = Node(prior_probability=normalized)
            node.children[move] = child
        node.is_expanded = True

    def _select_child(self, node: Node, unvisited_default: float | None = None) -> tuple[Move, Node]:
        """Select a child from the current player's perspective.

        Each child stores values for the *next* player to move, hence the
        negation for visited children.  First-play urgency is calculated from
        the parent value and is not injected into network targets.
        """
        parent_visits = float(max(1, node.visit_count))
        sqrt_parent_visit = math.sqrt(parent_visits)
        c_base = 19652.0
        c_init = float(self.c_puct)
        dynamic_cpuct = c_init + math.log((parent_visits + c_base + 1.0) / c_base)

        fpu_value = max(-1.0, min(1.0, node.q - self.fpu_reduction))
        best_score = -float("inf")
        best: tuple[Move, Node] | None = None
        for move, child in node.children.items():
            q_from_parent = opponent_value(child.q) if child.visit_count > 0 else fpu_value
            exploration = dynamic_cpuct * child.prior_probability * sqrt_parent_visit / (1 + child.visit_count)
            score = q_from_parent + exploration
            if score > best_score:
                best_score = score
                best = (move, child)
        if best is None:
            raise RuntimeError("cannot select from a node without children")
        return best

    def _puct_score(self, parent: Node, child: Node, unvisited_default: float | None = None) -> float:
        """Return the PUCT score used by :meth:`_select_child`.

        ``unvisited_default`` is accepted for source compatibility but FPU is
        the canonical unvisited estimate.
        """
        parent_visits = float(max(1, parent.visit_count))
        sqrt_parent_visit = math.sqrt(parent_visits)
        c_base = 19652.0
        c_init = float(self.c_puct)
        dynamic_cpuct = c_init + math.log((parent_visits + c_base + 1.0) / c_base)

        fpu_value = max(-1.0, min(1.0, parent.q - self.fpu_reduction))
        if unvisited_default is not None:
            fpu_value = float(unvisited_default)
        q_from_parent = opponent_value(child.q) if child.visit_count > 0 else fpu_value
        exploration = dynamic_cpuct * child.prior_probability * sqrt_parent_visit / (1 + child.visit_count)
        return q_from_parent + exploration

    def _backpropagate(self, path: list[Node], value: float, leaf_turn: int) -> None:
        """Back up a side-to-move value, negating exactly once per ply."""
        del leaf_turn  # Node depth, not colour constants, defines perspective.
        current_value = float(value)
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += current_value
            current_value = opponent_value(current_value)

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
            child.prior_probability = (
                1 - self.dirichlet_epsilon
            ) * child.prior_probability + self.dirichlet_epsilon * eta

    def _should_resign(self, board: Board, root: Node, generation: int) -> bool:
        """Optional explicit resignation policy; disabled by default in self-play."""
        if generation < 10 or self.resign_threshold <= -1.0:
            return False
        if root.q < self.resign_threshold:
            self._resign_streak[board.turn] = self._resign_streak.get(board.turn, 0) + 1
        else:
            self._resign_streak[board.turn] = 0
        return self._resign_streak.get(board.turn, 0) >= 10

    def advance_to(self, move: Move, history: list[Board] | None = None) -> None:
        if move in self.root.children:
            self.root = self.root.children[move]
            if self.root_board is not None and move in self.root_board.legal_moves():
                self.root_board = self.root_board.copy()
                self.root_board._push_unchecked(move)
            else:
                self.root_board = None
            self.root_hash = None
            self.root_context = _board_context(self.root_board, history) if self.root_board else None
            self._root_noise_applied = False
            self.transposition_table.clear()
            return
        self.root = Node()
        self.root_hash = None
        self.root_context = None
        self._root_noise_applied = False
        self.transposition_table.clear()


def _dirichlet(alphas: list[float], rng: random.Random) -> list[float]:
    samples = [rng.gammavariate(alpha, 1.0) for alpha in alphas]
    total = sum(samples)
    if total <= 0:
        return [1.0 / len(alphas)] * len(alphas)
    return [sample / total for sample in samples]


def _repetition_count(
    board: Board,
    history: list[Board],
    branch_positions: list[tuple[str, int, int, int | None]],
) -> int:
    """Count the current position across external and simulated history."""
    key = board._position_key()
    position_hash = board.zobrist_hash
    return (
        1
        + sum(position.zobrist_hash == position_hash and position._position_key() == key for position in history)
        + sum(position_key == key for position_key in branch_positions)
    )


def _board_context(board: Board, history: list[Board] | None = None) -> tuple[object, ...]:
    """Include neural-input and draw-history context in root reuse."""
    history_signature = tuple(
        (
            position._position_key(),
            position.halfmove_clock,
            position.fullmove_number,
        )
        for position in (history or [])
    )
    return (
        board._position_key(),
        board.halfmove_clock,
        board.fullmove_number,
        tuple(board.position_history),
        history_signature,
    )
