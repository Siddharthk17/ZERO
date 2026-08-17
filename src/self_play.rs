//! Parallel self-play orchestration. Implemented below the search/evaluator
//! modules so both Python and CLI callers use identical temperature and target
//! semantics.

use cozy_chess::{Board, GameStatus, Move};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::encoding::{move_to_policy_index, standard_uci, HistoryPosition};
use crate::evaluator::SharedGpuEvaluator;
use crate::mcts::{is_claimable_draw, is_dead_position, Mcts, MctsError, SearchConfig};

#[derive(Clone, Copy)]
pub struct SelfPlayConfig {
    /// Simulation budget for high-confidence PCR plies.
    pub simulations: u32,
    pub fast_simulations: u32,
    pub full_search_probability: f64,
    /// Relative confidence assigned to policy targets produced by fast PCR
    /// searches. Full-search targets always use weight `1.0`.
    pub fast_search_weight: f32,
    pub search_batch_size: usize,
    pub max_plies: usize,
    pub workers: usize,
    pub node_capacity: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            simulations: 400,
            fast_simulations: 35,
            full_search_probability: 0.20,
            fast_search_weight: 0.25,
            search_batch_size: 64,
            max_plies: 512,
            workers: 24,
            node_capacity: 250_000,
        }
    }
}

#[derive(Clone)]
pub struct Experience {
    pub fen: String,
    /// Sparse AlphaZero policy target: `(policy_index, probability)`. A chess
    /// position has at most 218 legal moves, so transferring a dense 4,672
    /// float vector through PyO3 would waste bandwidth and RAM.
    pub policy: Vec<(u16, f32)>,
    /// Confidence weight for the policy target produced by this search.
    /// Fast PCR positions remain useful training data, but their policy target
    /// is noisier than a full-search target.
    pub policy_weight: f32,
    pub value: f32,
    pub wdl: [f32; 3],
    /// Terminal targets are valid for WDL training; truncated targets are not.
    pub target_kind: &'static str,
    pub q_mcts: f32,
    pub material: (f32, f32),
    pub moves_left: f32,
    /// Occurrence count for `fen` at the time it was evaluated.
    pub repetitions: u8,
    /// Policy target for the immediately following ply, when it exists. Its
    /// coordinates use that opponent-to-move position's frame.
    pub opponent_policy: Option<Vec<(u16, f32)>>,
    pub opponent_legal_policy: Option<Vec<u16>>,
    pub history_fens: Vec<String>,
    /// Occurrence counts aligned with `history_fens`.
    pub history_repetitions: Vec<u8>,
}

#[derive(Clone)]
pub struct CompletedGame {
    pub result: f32,
    pub moves: Vec<String>,
    pub experiences: Vec<Experience>,
    pub termination: &'static str,
}

#[derive(Clone)]
struct PendingPosition {
    fen: String,
    history_fens: Vec<String>,
    history_repetitions: Vec<u8>,
    policy: Vec<(u16, f32)>,
    policy_weight: f32,
    legal_policy: Vec<u16>,
    q_mcts: f32,
    material: (f32, f32),
    repetitions: u8,
    side_to_move_is_white: bool,
    ply: usize,
}

pub fn generate_self_play(
    evaluator: Arc<SharedGpuEvaluator>,
    game_count: usize,
    config: SelfPlayConfig,
    seed: u64,
) -> Result<Vec<CompletedGame>, MctsError> {
    if config.simulations == 0 || config.fast_simulations == 0 {
        return Err(MctsError::InvalidConfig(
            "self-play simulation budgets must be positive",
        ));
    }
    if !config.full_search_probability.is_finite()
        || !(0.0..=1.0).contains(&config.full_search_probability)
    {
        return Err(MctsError::InvalidConfig(
            "full_search_probability must be in [0, 1]",
        ));
    }
    if !config.fast_search_weight.is_finite() || !(0.0..=1.0).contains(&config.fast_search_weight) {
        return Err(MctsError::InvalidConfig(
            "fast_search_weight must be finite and in [0, 1]",
        ));
    }
    if config.search_batch_size == 0 || config.search_batch_size > crate::mcts::MAX_BATCH {
        return Err(MctsError::BatchTooLarge);
    }
    if config.max_plies == 0 {
        return Err(MctsError::InvalidConfig("max_plies must be positive"));
    }
    if config.max_plies > crate::mcts::MAX_SEARCH_DEPTH {
        return Err(MctsError::InvalidConfig(
            "max_plies exceeds the fixed MCTS search depth",
        ));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.workers.max(1))
        .build()
        .map_err(|_| MctsError::InvalidTree)?;
    pool.install(|| {
        (0..game_count)
            .into_par_iter()
            .map(|index| {
                play_game(
                    Arc::clone(&evaluator),
                    config,
                    seed.wrapping_add(index as u64),
                )
            })
            .collect()
    })
}

fn play_game(
    evaluator: Arc<SharedGpuEvaluator>,
    config: SelfPlayConfig,
    seed: u64,
) -> Result<CompletedGame, MctsError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut board = Board::default();
    let mut mcts = Mcts::new(
        Arc::clone(&evaluator),
        config.node_capacity,
        config.search_batch_size,
    )?;
    let mut history = Vec::<HistoryPosition>::with_capacity(config.max_plies);
    let mut pending = Vec::<PendingPosition>::with_capacity(config.max_plies);
    let mut played_moves = Vec::with_capacity(config.max_plies);
    for ply in 0..config.max_plies {
        match board.status() {
            GameStatus::Won => {
                return finalize(pending, played_moves, board.side_to_move(), "checkmate")
            }
            GameStatus::Drawn => return finalize_draw(pending, played_moves, "draw"),
            GameStatus::Ongoing => {}
        }
        let repetitions = repetition_count(&board, &history);
        if is_claimable_draw(&board, &history, &[], repetitions)
            || is_dead_position(&board)
            || board.halfmove_clock() >= 100
        {
            return finalize_draw(pending, played_moves, "draw");
        }
        let is_full_search = rng.gen_bool(config.full_search_probability);
        let current_sims = if is_full_search {
            config.simulations
        } else {
            config.fast_simulations
        };

        let search = SearchConfig {
            simulations: current_sims,
            batch_size: config.search_batch_size,
            temperature: if ply < 12 { 1.0 } else { 0.05 },
            // AlphaZero exploration noise is mixed into the root for every
            // training move, not just the initial opening position.
            add_root_noise: true,
            ..SearchConfig::default()
        };
        // The temperature is sampled inside MCTS; no policy target is altered.
        let result = mcts.search(&board, &history, repetitions, search, &mut rng)?;
        let Some(chess_move) = result.best_move else {
            return finalize_draw(pending, played_moves, "no_legal_move");
        };
        // PCR retains every position. Fast searches use a reduced target
        // confidence rather than throwing away the generated game data.
        let policy = sparse_policy(&board, &result.visits);
        let legal_policy = legal_policy_indices(&board);
        let material = piece_material(&board);
        pending.push(PendingPosition {
            fen: board.to_string(),
            history_fens: history
                .iter()
                .take(7)
                .map(|entry| entry.board.to_string())
                .collect(),
            history_repetitions: history
                .iter()
                .take(7)
                .map(|entry| entry.repetitions)
                .collect(),
            policy,
            policy_weight: if is_full_search {
                1.0
            } else {
                config.fast_search_weight
            },
            legal_policy,
            q_mcts: result.root_value,
            material,
            repetitions,
            side_to_move_is_white: board.side_to_move() == cozy_chess::Color::White,
            ply,
        });
        played_moves.push(standard_uci(&board, chess_move));
        if history.len() == history.capacity() {
            // Game history is bounded by max_plies, so this only occurs if a
            // caller passes a pathological zero capacity (which Vec disallows).
            break;
        }
        history.insert(
            0,
            HistoryPosition {
                board: board.clone(),
                repetitions,
            },
        );
        board.play(chess_move);
        let next_repetitions = repetition_count(&board, &history);
        mcts.advance_to_with_context(chess_move, &board, &history, next_repetitions)?;
        match board.status() {
            GameStatus::Won => {
                return finalize(pending, played_moves, board.side_to_move(), "checkmate")
            }
            GameStatus::Drawn => return finalize_draw(pending, played_moves, "draw"),
            GameStatus::Ongoing
                if is_claimable_draw(&board, &history, &[], next_repetitions)
                    || is_dead_position(&board)
                    || board.halfmove_clock() >= 100 =>
            {
                return finalize_draw(pending, played_moves, "draw")
            }
            GameStatus::Ongoing => {}
        }
    }
    let final_status = board.status();
    match final_status {
        GameStatus::Won => finalize(pending, played_moves, board.side_to_move(), "checkmate"),
        GameStatus::Drawn => finalize_draw(pending, played_moves, "draw"),
        GameStatus::Ongoing => finalize_draw(pending, played_moves, "max_plies"),
    }
}

fn piece_material(board: &Board) -> (f32, f32) {
    let mut white_mat = 0.0;
    let mut black_mat = 0.0;
    for piece in [
        cozy_chess::Piece::Pawn,
        cozy_chess::Piece::Knight,
        cozy_chess::Piece::Bishop,
        cozy_chess::Piece::Rook,
        cozy_chess::Piece::Queen,
    ] {
        let val = match piece {
            cozy_chess::Piece::Pawn => 1.0,
            cozy_chess::Piece::Knight => 3.0,
            cozy_chess::Piece::Bishop => 3.0,
            cozy_chess::Piece::Rook => 5.0,
            cozy_chess::Piece::Queen => 9.0,
            _ => 0.0,
        };
        white_mat += board.colored_pieces(cozy_chess::Color::White, piece).len() as f32 * val;
        black_mat += board.colored_pieces(cozy_chess::Color::Black, piece).len() as f32 * val;
    }
    // Labels are normalized to the unit interval to keep this auxiliary MSE
    // commensurate with the policy and value losses.
    (white_mat / 39.0, black_mat / 39.0)
}

fn sparse_policy(board: &Board, visits: &[(Move, u32)]) -> Vec<(u16, f32)> {
    let total: u32 = visits.iter().map(|(_, count)| *count).sum();
    if total == 0 {
        let mut legal = Vec::with_capacity(218);
        board.generate_moves(|moves| {
            for chess_move in moves {
                if let Ok(index) = move_to_policy_index(board, chess_move) {
                    legal.push(index as u16);
                }
            }
            false
        });
        if legal.is_empty() {
            return legal.into_iter().map(|index| (index, 0.0)).collect();
        }
        let probability = 1.0 / legal.len() as f32;
        return legal
            .into_iter()
            .map(|index| (index, probability))
            .collect();
    }
    let mut policy = Vec::with_capacity(visits.len());
    for (chess_move, count) in visits {
        if *count == 0 {
            continue;
        }
        if let Ok(index) = move_to_policy_index(board, *chess_move) {
            policy.push((index as u16, *count as f32 / total as f32));
        }
    }
    policy
}

fn legal_policy_indices(board: &Board) -> Vec<u16> {
    let mut legal = Vec::with_capacity(218);
    board.generate_moves(|moves| {
        for chess_move in moves {
            if let Ok(index) = move_to_policy_index(board, chess_move) {
                legal.push(index as u16);
            }
        }
        false
    });
    legal
}

fn finalize(
    positions: Vec<PendingPosition>,
    moves: Vec<String>,
    losing_side: cozy_chess::Color,
    reason: &'static str,
) -> Result<CompletedGame, MctsError> {
    let white_value = if losing_side == cozy_chess::Color::White {
        -1.0
    } else {
        1.0
    };
    let total_plies = moves.len();
    let by_ply: HashMap<usize, usize> = positions
        .iter()
        .enumerate()
        .map(|(index, position)| (position.ply, index))
        .collect();
    let mut experiences = Vec::with_capacity(positions.len());
    for position in &positions {
        let value = if position.side_to_move_is_white {
            white_value
        } else {
            -white_value
        };
        let moves_left = normalized_moves_left(total_plies, position.ply);
        let opponent_position = by_ply
            .get(&(position.ply + 1))
            .and_then(|index| positions.get(*index));
        let opponent_policy = opponent_position.map(|candidate| candidate.policy.clone());
        let opponent_legal_policy =
            opponent_position.map(|candidate| candidate.legal_policy.clone());
        experiences.push(Experience {
            fen: position.fen.clone(),
            policy: position.policy.clone(),
            policy_weight: position.policy_weight,
            value,
            wdl: value_to_wdl(value),
            target_kind: "terminal",
            q_mcts: position.q_mcts,
            material: position.material,
            moves_left,
            repetitions: position.repetitions,
            opponent_policy,
            opponent_legal_policy,
            history_fens: position.history_fens.clone(),
            history_repetitions: position.history_repetitions.clone(),
        });
    }
    Ok(CompletedGame {
        result: white_value,
        moves,
        experiences,
        termination: reason,
    })
}

fn finalize_draw(
    positions: Vec<PendingPosition>,
    moves: Vec<String>,
    reason: &'static str,
) -> Result<CompletedGame, MctsError> {
    let total_plies = moves.len();
    let by_ply: HashMap<usize, usize> = positions
        .iter()
        .enumerate()
        .map(|(index, position)| (position.ply, index))
        .collect();
    let target_kind = if reason == "max_plies" {
        "truncated"
    } else {
        "terminal"
    };
    Ok(CompletedGame {
        result: 0.0,
        moves,
        experiences: positions
            .iter()
            .map(|position| {
                let moves_left = normalized_moves_left(total_plies, position.ply);
                let opponent_position = by_ply
                    .get(&(position.ply + 1))
                    .and_then(|index| positions.get(*index));
                let opponent_policy = opponent_position.map(|candidate| candidate.policy.clone());
                let opponent_legal_policy =
                    opponent_position.map(|candidate| candidate.legal_policy.clone());
                Experience {
                    fen: position.fen.clone(),
                    policy: position.policy.clone(),
                    policy_weight: position.policy_weight,
                    value: 0.0,
                    wdl: [0.0, 1.0, 0.0],
                    target_kind,
                    q_mcts: position.q_mcts,
                    material: position.material,
                    moves_left,
                    repetitions: position.repetitions,
                    opponent_policy,
                    opponent_legal_policy,
                    history_fens: position.history_fens.clone(),
                    history_repetitions: position.history_repetitions.clone(),
                }
            })
            .collect(),
        termination: reason,
    })
}

#[inline]
fn normalized_moves_left(total_plies: usize, ply: usize) -> f32 {
    total_plies.saturating_sub(ply).min(100) as f32 / 100.0
}

#[inline]
fn repetition_count(board: &Board, history: &[HistoryPosition]) -> u8 {
    let position_hash = board.hash_without_ep();
    history
        .iter()
        .filter(|entry| {
            entry.board.hash_without_ep() == position_hash && entry.board.same_position(board)
        })
        .count()
        .saturating_add(1)
        .min(u8::MAX as usize) as u8
}

#[inline]
fn value_to_wdl(value: f32) -> [f32; 3] {
    if value > 0.0 {
        [1.0, 0.0, 0.0]
    } else if value < 0.0 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn fast_search_weight_is_validated() {
        let evaluator = SharedGpuEvaluator::uniform(1, 1, Duration::ZERO);
        let config = SelfPlayConfig {
            fast_search_weight: f32::NAN,
            ..SelfPlayConfig::default()
        };
        let result = generate_self_play(evaluator.clone(), 0, config, 0);
        assert!(matches!(result, Err(MctsError::InvalidConfig(_))));
        evaluator.shutdown();
    }

    #[test]
    fn finalized_targets_preserve_policy_weight() {
        let position = PendingPosition {
            fen: "8/8/8/8/8/8/8/K6k w - - 0 1".to_owned(),
            history_fens: Vec::new(),
            history_repetitions: Vec::new(),
            policy: vec![(0, 1.0)],
            policy_weight: 0.25,
            legal_policy: vec![0],
            q_mcts: 0.0,
            material: (0.0, 0.0),
            repetitions: 1,
            side_to_move_is_white: true,
            ply: 0,
        };
        let game =
            finalize_draw(vec![position], Vec::new(), "max_plies").expect("target finalization");
        assert_eq!(game.experiences[0].policy_weight, 0.25);
        assert_eq!(game.experiences[0].target_kind, "truncated");
    }
}
