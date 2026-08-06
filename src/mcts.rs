//! Flat-arena, batched PUCT search.
//!
//! There is no `Box<Node>`, child `HashMap`, per-node lock, or pointer chase in
//! this implementation.  A node's children occupy a contiguous range in an
//! arena. Per-game trees are owned by one Rayon worker; CUDA work is shared by
//! `SharedGpuEvaluator`. Atomics make virtual loss/statistics race-free if the
//! selection front-end is later split across threads, without penalising the
//! normal single-owner fast path with mutexes.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use arrayvec::ArrayVec;
use cozy_chess::{Board, Color, GameStatus, Move, Piece};
use rand::Rng;

use crate::encoding::{encode_board_into, EncodedBoard, HistoryPosition, PolicyMask};
use crate::evaluator::{Evaluation, EvaluationError, EvaluationTicket, SharedGpuEvaluator};

pub const MAX_BATCH: usize = 256;
pub const MAX_SEARCH_DEPTH: usize = 512;
const INVALID_NODE: u32 = u32::MAX;

type NodeId = u32;

/// Atomic `f32` implemented as a CAS loop. This avoids an external lock when
/// virtual loss and backup touch a statistic shared by selection workers.
#[derive(Debug)]
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    #[inline]
    pub const fn new(value: f32) -> Self {
        Self(AtomicU32::new(value.to_bits()))
    }

    #[inline]
    pub fn load(&self, ordering: Ordering) -> f32 {
        f32::from_bits(self.0.load(ordering))
    }

    #[inline]
    pub fn store(&self, value: f32, ordering: Ordering) {
        self.0.store(value.to_bits(), ordering);
    }

    #[inline]
    pub fn fetch_add(&self, value: f32, ordering: Ordering) {
        let mut previous = self.0.load(Ordering::Relaxed);
        loop {
            let next = (f32::from_bits(previous) + value).to_bits();
            match self
                .0
                .compare_exchange_weak(previous, next, ordering, Ordering::Relaxed)
            {
                Ok(_) => return,
                Err(actual) => previous = actual,
            }
        }
    }
}

/// A compact node header. Child nodes carry the move from their parent; no move
/// key map is required because legal chess branching is small and contiguous.
#[derive(Debug)]
pub struct MctsNode {
    parent: NodeId,
    first_child: NodeId,
    child_count: u16,
    incoming_move: Option<Move>,
    prior: AtomicF32,
    visits: AtomicU32,
    value_sum: AtomicF32,
    // WDL is written exactly once, while this single-owner MCTS expands the
    // node.  Unlike visit/value statistics it is never updated by parallel
    // selection, so atomics would only add overhead here.
    wdl: [f32; 3],
    virtual_losses: AtomicU32,
    expanded: AtomicBool,
}

impl MctsNode {
    fn root() -> Self {
        Self::new(INVALID_NODE, None, 1.0)
    }

    fn new(parent: NodeId, incoming_move: Option<Move>, prior: f32) -> Self {
        Self {
            parent,
            first_child: INVALID_NODE,
            child_count: 0,
            incoming_move,
            prior: AtomicF32::new(prior),
            visits: AtomicU32::new(0),
            value_sum: AtomicF32::new(0.0),
            wdl: [0.0, 1.0, 0.0],
            virtual_losses: AtomicU32::new(0),
            expanded: AtomicBool::new(false),
        }
    }

    fn cloned_for_compaction(&self, parent: NodeId) -> Self {
        Self {
            parent,
            first_child: INVALID_NODE,
            child_count: 0,
            incoming_move: self.incoming_move,
            prior: AtomicF32::new(self.prior.load(Ordering::Relaxed)),
            visits: AtomicU32::new(self.visits.load(Ordering::Relaxed)),
            value_sum: AtomicF32::new(self.value_sum.load(Ordering::Relaxed)),
            wdl: self.wdl,
            virtual_losses: AtomicU32::new(0),
            expanded: AtomicBool::new(self.expanded.load(Ordering::Relaxed)),
        }
    }

    #[inline]
    fn q(&self) -> f32 {
        let visits = self.visits.load(Ordering::Relaxed);
        if visits == 0 {
            0.0
        } else {
            self.value_sum.load(Ordering::Relaxed) / visits as f32
        }
    }

    #[inline]
    fn apply_virtual_loss(&self) {
        self.visits.fetch_add(1, Ordering::Relaxed);
        self.value_sum.fetch_add(-1.0, Ordering::Relaxed);
        self.virtual_losses.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    fn undo_virtual_loss(&self) {
        if self
            .virtual_losses
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                count.checked_sub(1)
            })
            .is_ok()
        {
            self.visits.fetch_sub(1, Ordering::Relaxed);
            self.value_sum.fetch_add(1.0, Ordering::Relaxed);
        }
    }
}

struct Arena {
    nodes: Vec<MctsNode>,
    capacity: usize,
}

impl Arena {
    fn new(capacity: usize) -> Self {
        let capacity = capacity.max(2);
        let mut nodes = Vec::with_capacity(capacity);
        nodes.push(MctsNode::root());
        Self { nodes, capacity }
    }

    fn reset(&mut self) {
        self.nodes.clear();
        self.nodes.push(MctsNode::root());
    }

    #[inline]
    fn node(&self, id: NodeId) -> Option<&MctsNode> {
        self.nodes.get(id as usize)
    }

    fn reserve_children(&mut self, count: usize) -> Result<NodeId, MctsError> {
        if count == 0 || self.nodes.len().saturating_add(count) > self.capacity {
            return Err(MctsError::ArenaFull);
        }
        Ok(self.nodes.len() as NodeId)
    }

    fn push(&mut self, node: MctsNode) -> Result<NodeId, MctsError> {
        if self.nodes.len() >= self.capacity {
            return Err(MctsError::ArenaFull);
        }
        let id = self.nodes.len() as NodeId;
        self.nodes.push(node);
        Ok(id)
    }
}

#[derive(Debug)]
pub enum MctsError {
    ArenaFull,
    BatchTooLarge,
    InvalidConfig(&'static str),
    Evaluation(EvaluationError),
    InvalidTree,
}

impl fmt::Display for MctsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArenaFull => f.write_str("MCTS node arena is full"),
            Self::BatchTooLarge => write!(f, "MCTS batches may contain at most {MAX_BATCH} leaves"),
            Self::InvalidConfig(message) => write!(f, "invalid MCTS configuration: {message}"),
            Self::Evaluation(error) => write!(f, "evaluation error: {error}"),
            Self::InvalidTree => f.write_str("MCTS arena contains an invalid child range"),
        }
    }
}

impl std::error::Error for MctsError {}

impl From<EvaluationError> for MctsError {
    fn from(value: EvaluationError) -> Self {
        Self::Evaluation(value)
    }
}

#[derive(Clone, Copy)]
pub struct SearchConfig {
    pub simulations: u32,
    pub batch_size: usize,
    pub c_puct: f32,
    pub fpu_reduction: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub add_root_noise: bool,
    pub temperature: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 800,
            batch_size: 64,
            c_puct: 1.25,
            fpu_reduction: 0.25,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            add_root_noise: true,
            temperature: 1.0,
        }
    }
}

#[derive(Clone)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub visits: ArrayVec<(Move, u32), 256>,
    pub root_value: f32,
    pub root_wdl: [f32; 3],
}

struct Leaf {
    node: NodeId,
    board: Board,
    path: ArrayVec<NodeId, MAX_SEARCH_DEPTH>,
    encoded: EncodedBoard,
    legal_moves: ArrayVec<Move, 256>,
    legal_mask: PolicyMask,
}

/// One MCTS tree is deliberately confined to one self-play worker. That gives
/// excellent cache locality and avoids lock contention; simultaneous games
/// still coalesce their leaves in the shared CUDA actor.
pub struct Mcts {
    evaluator: Arc<SharedGpuEvaluator>,
    arena: Arena,
    spare_arena: Arena,
    root: NodeId,
    root_board: Option<Board>,
    root_context: Option<u64>,
    root_noise_applied: bool,
    leaves: Vec<Leaf>,
    tickets: Vec<EvaluationTicket>,
    compact_queue: Vec<NodeId>,
    remap: Vec<NodeId>,
}

impl Mcts {
    pub fn new(
        evaluator: Arc<SharedGpuEvaluator>,
        node_capacity: usize,
        batch_capacity: usize,
    ) -> Result<Self, MctsError> {
        if batch_capacity == 0 || batch_capacity > MAX_BATCH {
            return Err(MctsError::BatchTooLarge);
        }
        let node_capacity = node_capacity.max(batch_capacity + 1);
        Ok(Self {
            evaluator,
            arena: Arena::new(node_capacity),
            spare_arena: Arena::new(node_capacity),
            root: 0,
            root_board: None,
            root_context: None,
            root_noise_applied: false,
            leaves: Vec::with_capacity(batch_capacity),
            tickets: Vec::with_capacity(batch_capacity),
            compact_queue: Vec::with_capacity(node_capacity),
            remap: vec![INVALID_NODE; node_capacity],
        })
    }

    pub fn reset(&mut self) {
        self.arena.reset();
        self.spare_arena.reset();
        self.root = 0;
        self.root_board = None;
        self.root_context = None;
        self.root_noise_applied = false;
        self.leaves.clear();
        self.tickets.clear();
    }

    /// Run a batched PUCT search. `history` is newest-first and excludes
    /// `board`; it carries the exact history planes and repetition flags.
    pub fn search<R: Rng + ?Sized>(
        &mut self,
        board: &Board,
        history: &[HistoryPosition],
        current_repetitions: u8,
        config: SearchConfig,
        rng: &mut R,
    ) -> Result<SearchResult, MctsError> {
        if config.batch_size == 0
            || config.batch_size > self.leaves.capacity()
            || config.batch_size > MAX_BATCH
        {
            return Err(MctsError::BatchTooLarge);
        }
        validate_config(config)?;
        self.ensure_root(board, history, current_repetitions);
        let status = board.status();
        if matches!(status, GameStatus::Won) {
            let value = terminal_value(status);
            return Ok(SearchResult {
                best_move: None,
                visits: ArrayVec::new(),
                root_value: value,
                root_wdl: terminal_wdl(value),
            });
        }
        if current_repetitions >= 3
            || is_dead_position(board)
            || board.halfmove_clock() >= 100
            || matches!(status, GameStatus::Drawn)
        {
            return Ok(SearchResult {
                best_move: None,
                visits: ArrayVec::new(),
                root_value: 0.0,
                root_wdl: terminal_wdl(0.0),
            });
        }

        if !self.node(self.root)?.expanded.load(Ordering::Acquire) {
            self.expand_root(board, history, current_repetitions)?;
        }
        if config.add_root_noise && !self.root_noise_applied {
            self.add_dirichlet_noise(config.dirichlet_alpha, config.dirichlet_epsilon, rng)?;
            self.root_noise_applied = true;
        }

        let mut done = 0_u32;
        while done < config.simulations {
            let want = ((config.simulations - done) as usize).min(config.batch_size);
            let completed_terminals =
                self.collect_leaves(board, history, current_repetitions, want, config)?;
            done = done.saturating_add(completed_terminals as u32);
            if self.leaves.is_empty() {
                if completed_terminals == 0 {
                    break;
                }
                continue;
            }
            self.tickets.clear();
            let mut submit_error = None;
            for leaf in &self.leaves {
                match self.evaluator.submit(&leaf.encoded, leaf.legal_mask) {
                    Ok(ticket) => self.tickets.push(ticket),
                    Err(error) => {
                        submit_error = Some(MctsError::from(error));
                        break;
                    }
                }
            }
            if let Some(error) = submit_error {
                for ticket in self.tickets.drain(..) {
                    let _ = ticket.wait();
                }
                for leaf in &self.leaves {
                    self.undo_virtual_loss(&leaf.path)?;
                }
                self.leaves.clear();
                return Err(error);
            }
            let mut evaluation_error = None;
            while let (Some(leaf), Some(ticket)) = (self.leaves.pop(), self.tickets.pop()) {
                self.undo_virtual_loss(&leaf.path)?;
                match ticket.wait() {
                    Ok(result) => {
                        if let Err(error) = self.expand_with_evaluation(
                            leaf.node,
                            &leaf.board,
                            &result,
                            Some(&leaf.legal_moves),
                        ) {
                            evaluation_error = Some(error);
                            break;
                        }
                        if let Err(error) = self.backup(&leaf.path, result.value) {
                            evaluation_error = Some(error);
                            break;
                        }
                        done = done.saturating_add(1);
                    }
                    Err(error) => {
                        evaluation_error = Some(MctsError::from(error));
                        break;
                    }
                }
            }
            if let Some(error) = evaluation_error {
                for leaf in &self.leaves {
                    self.undo_virtual_loss(&leaf.path)?;
                }
                self.leaves.clear();
                self.tickets.clear();
                return Err(error);
            }
        }

        let visits = self.root_visits()?;
        let best_move = select_temperature_move(&visits, config.temperature, rng);
        let root = self.node(self.root)?;
        Ok(SearchResult {
            best_move,
            visits,
            root_value: root.q(),
            root_wdl: root.wdl,
        })
    }

    /// Make a selected move the root.  The reachable subtree is copied into a
    /// preallocated spare arena, re-indexed from zero, then the arena vectors
    /// are swapped. Thus old siblings are immediately reclaimable and tree
    /// reuse cannot exhaust the fixed node pool over a 512-ply game.
    pub fn advance_to(&mut self, played: Move, next_board: &Board) -> Result<(), MctsError> {
        self.advance_to_internal(played, next_board)?;
        // Callers that do not carry history still get safe behaviour: the
        // next search will invalidate the reused root when its context is
        // supplied to `ensure_root`.
        self.root_context = None;
        Ok(())
    }

    /// Advance the tree while preserving the history-dependent input context.
    pub fn advance_to_with_context(
        &mut self,
        played: Move,
        next_board: &Board,
        history: &[HistoryPosition],
        repetitions: u8,
    ) -> Result<(), MctsError> {
        self.advance_to_internal(played, next_board)?;
        self.root_context = Some(context_signature(next_board, history, repetitions));
        Ok(())
    }

    fn advance_to_internal(&mut self, played: Move, next_board: &Board) -> Result<(), MctsError> {
        let root = self.node(self.root)?;
        let mut next = None;
        for child_id in child_ids(root) {
            let child = self.node(child_id)?;
            if child.parent == self.root && child.incoming_move == Some(played) {
                next = Some(child_id);
                break;
            }
        }
        if let Some(next) = next {
            if self.compact_from(next).is_err() {
                self.arena.reset();
                self.root = 0;
            } else {
                self.root = 0;
            }
        } else {
            self.arena.reset();
            self.root = 0;
        }
        self.root_board = Some(next_board.clone());
        self.root_noise_applied = false;
        Ok(())
    }

    fn ensure_root(&mut self, board: &Board, history: &[HistoryPosition], repetitions: u8) {
        let context = context_signature(board, history, repetitions);
        if self.root_board.as_ref() != Some(board) || self.root_context != Some(context) {
            self.arena.reset();
            self.root = 0;
            self.root_board = Some(board.clone());
            self.root_context = Some(context);
            self.root_noise_applied = false;
        }
    }

    #[inline]
    fn node(&self, id: NodeId) -> Result<&MctsNode, MctsError> {
        self.arena.node(id).ok_or(MctsError::InvalidTree)
    }

    fn expand_root(
        &mut self,
        board: &Board,
        history: &[HistoryPosition],
        repetitions: u8,
    ) -> Result<(), MctsError> {
        let mut encoded = [0.0; crate::encoding::INPUT_SIZE];
        encode_board_into(board, history, repetitions, &mut encoded);
        let (_, legal_mask) = legal_moves_and_mask(board);
        let eval = self.evaluator.evaluate(&encoded, legal_mask)?;
        self.expand_with_evaluation(self.root, board, &eval, None)
    }

    fn collect_leaves(
        &mut self,
        root_board: &Board,
        root_history: &[HistoryPosition],
        root_repetitions: u8,
        count: usize,
        config: SearchConfig,
    ) -> Result<usize, MctsError> {
        self.leaves.clear();
        let mut terminal_count = 0;
        for _ in 0..count {
            let mut board = root_board.clone();
            let mut node = self.root;
            let mut path = ArrayVec::<NodeId, MAX_SEARCH_DEPTH>::new();
            // Encoding only consumes the most recent seven prior positions,
            // but threefold-repetition detection must see the *entire* game
            // history.  Keeping those concerns separate prevents an old
            // repetition from being silently missed during search.
            let mut history = ArrayVec::<HistoryPosition, { crate::encoding::HISTORY - 1 }>::new();
            for entry in root_history.iter().take(crate::encoding::HISTORY - 1) {
                let _ = history.try_push(entry.clone());
            }
            let mut branch_history = ArrayVec::<Board, MAX_SEARCH_DEPTH>::new();
            let mut repetitions = root_repetitions;
            path.push(node);

            loop {
                let current = self.node(node)?;
                if !current.expanded.load(Ordering::Acquire) || current.child_count == 0 {
                    break;
                }
                let selected = self.select_child(node, config)?;
                let child = self.node(selected)?;
                let Some(chess_move) = child.incoming_move else {
                    return Err(MctsError::InvalidTree);
                };
                let previous = HistoryPosition {
                    board: board.clone(),
                    repetitions,
                };
                if history.len() == history.capacity() {
                    let _ = history.pop();
                }
                history.insert(0, previous);
                if branch_history.try_push(board.clone()).is_err() {
                    self.backup(&path, 0.0)?;
                    terminal_count += 1;
                    break;
                }
                board.play(chess_move);
                repetitions = count_repetitions(&board, root_history, &branch_history);
                node = selected;
                if path.try_push(node).is_err() {
                    // A pathological line is treated as a neutral leaf rather
                    // than risking a stack allocation or hot-loop panic.
                    self.backup(&path, 0.0)?;
                    terminal_count += 1;
                    break;
                }
            }
            if path.last().copied() != Some(node) {
                continue;
            }
            match board.status() {
                GameStatus::Ongoing
                    if repetitions < 3
                        && !is_dead_position(&board)
                        && board.halfmove_clock() < 100 =>
                {
                    self.apply_virtual_loss(&path)?;
                    let mut encoded = [0.0; crate::encoding::INPUT_SIZE];
                    encode_board_into(&board, &history, repetitions, &mut encoded);
                    let (legal_moves, legal_mask) = legal_moves_and_mask(&board);
                    self.leaves.push(Leaf {
                        node,
                        board,
                        path,
                        encoded,
                        legal_moves,
                        legal_mask,
                    });
                }
                status => {
                    let value = match status {
                        GameStatus::Won => terminal_value(status),
                        GameStatus::Drawn | GameStatus::Ongoing if repetitions >= 3 => 0.0,
                        GameStatus::Drawn | GameStatus::Ongoing => terminal_value(status),
                    };
                    self.backup(&path, value)?;
                    terminal_count += 1;
                }
            }
        }
        Ok(terminal_count)
    }

    fn expand_with_evaluation(
        &mut self,
        id: NodeId,
        board: &Board,
        eval: &Evaluation,
        legal_moves: Option<&[Move]>,
    ) -> Result<(), MctsError> {
        if self.node(id)?.expanded.load(Ordering::Acquire) {
            return Ok(());
        }
        let generated_moves = legal_moves.map_or_else(
            || legal_moves_and_mask(board).0.to_vec(),
            |moves| moves.to_vec(),
        );
        let legal_count = generated_moves.len();
        let mut prior_sum = 0.0_f32;
        for chess_move in &generated_moves {
            if let Ok(index) = crate::encoding::move_to_policy_index(board, *chess_move) {
                prior_sum += eval.policy[index].max(0.0);
            }
        }
        if legal_count == 0 {
            if let Some(node) = self.arena.nodes.get(id as usize) {
                node.expanded.store(true, Ordering::Release);
            }
            return Ok(());
        }
        let first = self.arena.reserve_children(legal_count)?;
        let fallback = 1.0 / legal_count as f32;
        let mut added = 0_usize;
        for chess_move in generated_moves {
            let prior = crate::encoding::move_to_policy_index(board, chess_move)
                .ok()
                .map(|index| {
                    if prior_sum > 0.0 {
                        eval.policy[index].max(0.0) / prior_sum
                    } else {
                        fallback
                    }
                })
                .unwrap_or(fallback);
            if self
                .arena
                .push(MctsNode::new(id, Some(chess_move), prior))
                .is_ok()
            {
                added += 1;
            }
        }
        let node = self
            .arena
            .nodes
            .get_mut(id as usize)
            .ok_or(MctsError::InvalidTree)?;
        node.wdl = normalized_wdl(eval.wdl);
        node.first_child = first;
        node.child_count = added.min(u16::MAX as usize) as u16;
        node.expanded.store(true, Ordering::Release);
        Ok(())
    }

    fn select_child(&self, parent_id: NodeId, config: SearchConfig) -> Result<NodeId, MctsError> {
        let parent = self.node(parent_id)?;
        let parent_visits = parent.visits.load(Ordering::Relaxed).max(1) as f32;
        let sqrt_parent = parent_visits.sqrt();

        // Dynamic c_puct calculation (AlphaZero / KataGo formula)
        let c_base = 19652.0_f32;
        let c_init = config.c_puct;
        let dynamic_cpuct = c_init + ((parent_visits + c_base + 1.0) / c_base).ln();

        let fpu = (parent.q() - config.fpu_reduction).clamp(-1.0, 1.0);
        let mut best: Option<(NodeId, f32)> = None;
        for child_id in child_ids(parent) {
            let child = self.node(child_id)?;
            let visits = child.visits.load(Ordering::Relaxed);
            let q_from_parent = if visits == 0 { fpu } else { -child.q() };
            let score = q_from_parent
                + dynamic_cpuct * child.prior.load(Ordering::Relaxed) * sqrt_parent
                    / (1 + visits) as f32;
            if best
                .map(|(_, best_score)| score > best_score)
                .unwrap_or(true)
            {
                best = Some((child_id, score));
            }
        }
        best.map(|(id, _)| id).ok_or(MctsError::InvalidTree)
    }

    fn apply_virtual_loss(&self, path: &[NodeId]) -> Result<(), MctsError> {
        for &id in path {
            self.node(id)?.apply_virtual_loss();
        }
        Ok(())
    }

    fn undo_virtual_loss(&self, path: &[NodeId]) -> Result<(), MctsError> {
        for &id in path {
            self.node(id)?.undo_virtual_loss();
        }
        Ok(())
    }

    fn backup(&self, path: &[NodeId], leaf_value: f32) -> Result<(), MctsError> {
        let mut value = leaf_value;
        for &id in path.iter().rev() {
            let node = self.node(id)?;
            node.visits.fetch_add(1, Ordering::Relaxed);
            node.value_sum.fetch_add(value, Ordering::Relaxed);
            value = -value;
        }
        Ok(())
    }

    fn add_dirichlet_noise<R: Rng + ?Sized>(
        &self,
        alpha: f32,
        epsilon: f32,
        rng: &mut R,
    ) -> Result<(), MctsError> {
        let root = self.node(self.root)?;
        if root.child_count == 0 || alpha <= 0.0 || epsilon <= 0.0 {
            return Ok(());
        }
        let mut total = 0.0_f32;
        // Store samples in the node's temporary prior field after preserving
        // original priors in a local calculation on the second loop would need
        // memory. Instead compute a normalized gamma sample by a deterministic
        // two-pass replay of a per-call seed-free stream is impossible. A small
        // preallocated vector would add no nodes, but the hot path avoids it by
        // using an arena-local remap scratch prefix as f32 bit storage.
        // Here we use the spare remap capacity as an integer sample order is
        // not viable, so retain a fixed stack array bounded by chess moves.
        let mut samples = ArrayVec::<f32, 256>::new();
        for _ in child_ids(root) {
            let sample = sample_gamma(alpha, rng);
            total += sample;
            let _ = samples.try_push(sample);
        }
        if total <= f32::EPSILON {
            return Ok(());
        }
        for (offset, child_id) in child_ids(root).enumerate() {
            let child = self.node(child_id)?;
            let old = child.prior.load(Ordering::Relaxed);
            let noise = samples[offset] / total;
            child
                .prior
                .store((1.0 - epsilon) * old + epsilon * noise, Ordering::Relaxed);
        }
        Ok(())
    }

    fn root_visits(&self) -> Result<ArrayVec<(Move, u32), 256>, MctsError> {
        let root = self.node(self.root)?;
        let mut visits = ArrayVec::<(Move, u32), 256>::new();
        for child_id in child_ids(root) {
            let child = self.node(child_id)?;
            if let Some(chess_move) = child.incoming_move {
                let _ = visits.try_push((chess_move, child.visits.load(Ordering::Relaxed)));
            }
        }
        Ok(visits)
    }

    fn compact_from(&mut self, old_root: NodeId) -> Result<(), MctsError> {
        if self.arena.node(old_root).is_none() {
            return Err(MctsError::InvalidTree);
        }
        self.spare_arena.nodes.clear();
        self.remap.fill(INVALID_NODE);
        self.compact_queue.clear();
        let old = self.node(old_root)?;
        self.spare_arena
            .push(old.cloned_for_compaction(INVALID_NODE))?;
        self.remap[old_root as usize] = 0;
        self.compact_queue.push(old_root);

        let mut cursor = 0;
        while cursor < self.compact_queue.len() {
            let old_parent_id = self.compact_queue[cursor];
            cursor += 1;
            let new_parent_id = self.remap[old_parent_id as usize];
            let old_parent = self.node(old_parent_id)?;
            let old_children: ArrayVec<NodeId, 256> = child_ids(old_parent).collect();
            let first = self.spare_arena.nodes.len() as NodeId;
            let mut copied = 0_usize;
            for old_child_id in old_children {
                let old_child = self.node(old_child_id)?;
                let new_child = self
                    .spare_arena
                    .push(old_child.cloned_for_compaction(new_parent_id))?;
                self.remap[old_child_id as usize] = new_child;
                self.compact_queue.push(old_child_id);
                copied += 1;
            }
            let new_parent = self
                .spare_arena
                .nodes
                .get_mut(new_parent_id as usize)
                .ok_or(MctsError::InvalidTree)?;
            new_parent.first_child = if copied == 0 { INVALID_NODE } else { first };
            new_parent.child_count = copied as u16;
        }
        std::mem::swap(&mut self.arena.nodes, &mut self.spare_arena.nodes);
        Ok(())
    }
}

fn validate_config(config: SearchConfig) -> Result<(), MctsError> {
    if !config.c_puct.is_finite() || config.c_puct < 0.0 {
        return Err(MctsError::InvalidConfig(
            "c_puct must be finite and non-negative",
        ));
    }
    if !config.fpu_reduction.is_finite() || config.fpu_reduction < 0.0 {
        return Err(MctsError::InvalidConfig(
            "fpu_reduction must be finite and non-negative",
        ));
    }
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err(MctsError::InvalidConfig(
            "temperature must be finite and non-negative",
        ));
    }
    if config.add_root_noise
        && (!config.dirichlet_alpha.is_finite()
            || config.dirichlet_alpha <= 0.0
            || !config.dirichlet_epsilon.is_finite()
            || !(0.0..=1.0).contains(&config.dirichlet_epsilon))
    {
        return Err(MctsError::InvalidConfig(
            "root-noise alpha must be positive and epsilon must be in [0, 1]",
        ));
    }
    Ok(())
}

fn child_ids(node: &MctsNode) -> impl Iterator<Item = NodeId> + '_ {
    let first = node.first_child;
    let count = node.child_count as NodeId;
    (0..count).map(move |offset| first.saturating_add(offset))
}

fn legal_moves_and_mask(board: &Board) -> (ArrayVec<Move, 256>, PolicyMask) {
    let mut moves = ArrayVec::<Move, 256>::new();
    let mut mask = PolicyMask::default();
    board.generate_moves(|generated| {
        for chess_move in generated {
            if let Ok(index) = crate::encoding::move_to_policy_index(board, chess_move) {
                let _ = moves.try_push(chess_move);
                mask.set(index);
            }
        }
        false
    });
    (moves, mask)
}

/// Return whether the position is a dead position under the material rule.
/// `cozy-chess` intentionally leaves this rule to callers because it is not
/// needed for legal move generation.
pub fn is_dead_position(board: &Board) -> bool {
    if [Piece::Pawn, Piece::Rook, Piece::Queen]
        .iter()
        .any(|piece| !board.pieces(*piece).is_empty())
    {
        return false;
    }

    let mut bishop_square_colors = [Vec::new(), Vec::new()];
    let mut knights = [0_usize; 2];
    for color in [Color::White, Color::Black] {
        let color_index = color as usize;
        for square in board.colored_pieces(color, Piece::Bishop) {
            let index = square as usize;
            bishop_square_colors[color_index].push(((index & 7) + (index >> 3)) & 1);
        }
        knights[color_index] = board.colored_pieces(color, Piece::Knight).len() as usize;
    }

    let bishop_count = bishop_square_colors[0].len() + bishop_square_colors[1].len();
    let minor_count = bishop_count + knights[0] + knights[1];
    if minor_count <= 1 {
        return true;
    }
    if knights != [0, 0] {
        return false;
    }
    let mut bishop_color = None;
    for color in bishop_square_colors.iter().flatten().copied() {
        if bishop_color.is_some_and(|previous| previous != color) {
            return false;
        }
        bishop_color = Some(color);
    }
    true
}

fn context_signature(board: &Board, history: &[HistoryPosition], repetitions: u8) -> u64 {
    let mut signature = board.hash().wrapping_add(repetitions as u64);
    for (index, entry) in history.iter().enumerate() {
        signature = signature.rotate_left(7)
            ^ entry
                .board
                .hash()
                .wrapping_add((entry.repetitions as u64) << (index % 8));
    }
    signature
}

#[inline]
fn count_repetitions(
    board: &Board,
    root_history: &[HistoryPosition],
    branch_history: &[Board],
) -> u8 {
    (1 + root_history
        .iter()
        .filter(|entry| entry.board.same_position(board))
        .count()
        + branch_history
            .iter()
            .filter(|entry| entry.same_position(board))
            .count())
    .min(u8::MAX as usize) as u8
}

#[inline]
fn normalized_wdl(wdl: [f32; 3]) -> [f32; 3] {
    let sanitized = wdl.map(|value| {
        if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        }
    });
    let total = sanitized.iter().sum::<f32>();
    if total > f32::EPSILON {
        sanitized.map(|value| value / total)
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[inline]
fn terminal_value(status: GameStatus) -> f32 {
    match status {
        GameStatus::Won => -1.0, // side to move has been checkmated
        GameStatus::Drawn | GameStatus::Ongoing => 0.0,
    }
}

#[inline]
fn terminal_wdl(value: f32) -> [f32; 3] {
    if value > 0.0 {
        [1.0, 0.0, 0.0]
    } else if value < 0.0 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn select_temperature_move<R: Rng + ?Sized>(
    visits: &[(Move, u32)],
    temperature: f32,
    rng: &mut R,
) -> Option<Move> {
    if visits.is_empty() {
        return None;
    }
    if temperature <= 1.0e-6 {
        return visits
            .iter()
            .max_by_key(|(_, count)| count)
            .map(|(mv, _)| *mv);
    }
    let inverse_temperature = 1.0 / temperature;
    let total: f64 = visits
        .iter()
        .map(|(_, count)| (*count as f64).powf(inverse_temperature as f64))
        .sum();
    if !total.is_finite() || total <= 0.0 {
        return visits
            .iter()
            .max_by_key(|(_, count)| count)
            .map(|(mv, _)| *mv);
    }
    let mut sample = rng.gen::<f64>() * total;
    for (mv, count) in visits {
        sample -= (*count as f64).powf(inverse_temperature as f64);
        if sample <= 0.0 {
            return Some(*mv);
        }
    }
    visits.last().map(|(mv, _)| *mv)
}

/// Marsaglia-Tsang gamma sampler, allocation-free and accurate for the small
/// AlphaZero root-noise alpha (normally 0.3).
fn sample_gamma<R: Rng + ?Sized>(alpha: f32, rng: &mut R) -> f32 {
    if alpha <= 0.0 {
        return 0.0;
    }
    if alpha < 1.0 {
        return sample_gamma(alpha + 1.0, rng)
            * rng.gen::<f32>().max(f32::MIN_POSITIVE).powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = (1.0 / (9.0 * d)).sqrt();
    loop {
        let u1 = rng.gen::<f32>().max(f32::MIN_POSITIVE);
        let u2 = rng.gen::<f32>();
        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        let v = 1.0 + c * normal;
        if v <= 0.0 {
            continue;
        }
        let cubed = v * v * v;
        let uniform = rng.gen::<f32>().max(f32::MIN_POSITIVE);
        if uniform < 1.0 - 0.0331 * normal.powi(4)
            || uniform.ln() < 0.5 * normal * normal + d * (1.0 - cubed + cubed.ln())
        {
            return d * cubed;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn uniform_search_reuses_a_compacted_subtree() {
        let evaluator = SharedGpuEvaluator::uniform(16, 16, Duration::from_micros(50));
        let mut mcts = Mcts::new(Arc::clone(&evaluator), 8_192, 8).expect("valid fixed capacities");
        let mut board = Board::default();
        let mut rng = rand::thread_rng();
        let result = mcts
            .search(
                &board,
                &[],
                1,
                SearchConfig {
                    simulations: 16,
                    batch_size: 8,
                    add_root_noise: false,
                    temperature: 0.0,
                    ..SearchConfig::default()
                },
                &mut rng,
            )
            .expect("uniform search should complete");
        let chess_move = result.best_move.expect("start position has legal moves");
        board.play(chess_move);
        mcts.advance_to(chess_move, &board)
            .expect("child should become the root");
        let follow_up = mcts
            .search(
                &board,
                &[],
                1,
                SearchConfig {
                    simulations: 8,
                    batch_size: 8,
                    add_root_noise: false,
                    temperature: 0.0,
                    ..SearchConfig::default()
                },
                &mut rng,
            )
            .expect("compacted subtree should remain searchable");
        assert!(follow_up.best_move.is_some());
        evaluator.shutdown();
    }

    #[test]
    fn repetition_count_uses_full_game_history() {
        let board = Board::default();
        let history = vec![
            HistoryPosition {
                board: board.clone(),
                repetitions: 1,
            };
            8
        ];
        assert_eq!(count_repetitions(&board, &history, &[]), 9);
    }

    #[test]
    fn repetition_identity_ignores_move_counters() {
        let with_counters: Board = "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1"
            .parse()
            .expect("valid board");
        let without_counters: Board = "4k3/8/8/8/8/8/4P3/4K3 b - - 42 20"
            .parse()
            .expect("valid board");
        assert!(with_counters.same_position(&without_counters));
        assert_eq!(
            count_repetitions(
                &with_counters,
                &[HistoryPosition {
                    board: without_counters,
                    repetitions: 1,
                }],
                &[],
            ),
            2
        );
    }

    #[test]
    fn dead_material_is_terminal() {
        let kings: Board = "8/8/8/8/8/8/8/K6k w - - 0 1".parse().expect("valid board");
        let rook: Board = "7k/8/8/8/8/8/8/R5K1 w - - 0 1"
            .parse()
            .expect("valid board");
        assert!(is_dead_position(&kings));
        assert!(!is_dead_position(&rook));
    }

    #[test]
    fn same_color_bishops_are_dead_material() {
        let board: Board = "7k/8/8/8/5b2/8/8/K1B5 w - - 0 1"
            .parse()
            .expect("valid same-color bishop position");
        assert!(is_dead_position(&board));
    }

    #[test]
    fn opposite_color_bishops_are_not_dead_material() {
        let board: Board = "7k/8/8/8/4b3/8/8/K1B5 w - - 0 1"
            .parse()
            .expect("valid opposite-color bishop position");
        assert!(!is_dead_position(&board));
    }

    #[test]
    fn wdl_is_normalized_and_has_a_safe_fallback() {
        assert_eq!(normalized_wdl([2.0, 1.0, 1.0]), [0.5, 0.25, 0.25]);
        assert_eq!(normalized_wdl([f32::NAN, -1.0, 0.0]), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn invalid_search_parameters_fail_before_selection() {
        let invalid = SearchConfig {
            temperature: f32::NAN,
            ..SearchConfig::default()
        };
        assert!(matches!(
            validate_config(invalid),
            Err(MctsError::InvalidConfig(_))
        ));
    }
}
