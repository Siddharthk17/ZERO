//! UCI binary entry point. Model-backed analysis is exposed through the PyO3
//! extension; this executable retains a legal native uniform-MCTS fallback for
//! protocol smoke tests and infrastructure that requires a standalone engine.

use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::time::Duration;

use cozy_chess::{Board, Move};
use zero_rust_engine::encoding::{standard_uci, HistoryPosition};
use zero_rust_engine::evaluator::SharedGpuEvaluator;
use zero_rust_engine::mcts::{Mcts, SearchConfig};

struct UciSession {
    board: Board,
    history: Vec<HistoryPosition>,
    evaluator: Arc<SharedGpuEvaluator>,
    mcts: Mcts,
}

impl UciSession {
    fn new() -> Result<Self, String> {
        let evaluator = SharedGpuEvaluator::uniform(256, 256, Duration::from_micros(1_000));
        let mcts =
            Mcts::new(Arc::clone(&evaluator), 250_000, 64).map_err(|error| error.to_string())?;
        Ok(Self {
            board: Board::default(),
            history: Vec::with_capacity(512),
            evaluator,
            mcts,
        })
    }

    fn set_position(&mut self, command: &str) -> Result<(), String> {
        let words: Vec<_> = command.split_whitespace().collect();
        let moves_at = words.iter().position(|word| *word == "moves");
        let position_words = &words[..moves_at.unwrap_or(words.len())];
        let mut candidate_board = if position_words.get(1) == Some(&"startpos") {
            Board::default()
        } else if position_words.get(1) == Some(&"fen") && position_words.len() >= 8 {
            position_words[2..8]
                .join(" ")
                .parse::<Board>()
                .map_err(|error| error.to_string())?
        } else {
            return Err("invalid position command".to_owned());
        };
        let mut candidate_history = Vec::with_capacity(512);
        if let Some(index) = moves_at {
            for uci in &words[index + 1..] {
                let chess_move = legal_uci_move(&candidate_board, uci)
                    .ok_or_else(|| format!("illegal move in position command: {uci}"))?;
                let repetitions = repetitions(&candidate_board, &candidate_history);
                candidate_history.insert(
                    0,
                    HistoryPosition {
                        board: candidate_board.clone(),
                        repetitions,
                    },
                );
                candidate_board.play(chess_move);
            }
        }
        self.board = candidate_board;
        self.history = candidate_history;
        self.mcts.reset();
        Ok(())
    }

    fn new_game(&mut self) {
        self.board = Board::default();
        self.history.clear();
        self.mcts.reset();
    }

    fn go(&mut self, command: &str) -> Result<Option<Move>, String> {
        let words: Vec<_> = command.split_whitespace().collect();
        let simulations = words
            .windows(2)
            .find_map(|pair| {
                (pair[0] == "nodes")
                    .then(|| pair[1].parse::<u32>().ok())
                    .flatten()
            })
            .unwrap_or(800)
            .clamp(1, 50_000);
        let result = self
            .mcts
            .search(
                &self.board,
                &self.history,
                repetitions(&self.board, &self.history),
                SearchConfig {
                    simulations,
                    batch_size: 64,
                    temperature: 0.0,
                    add_root_noise: false,
                    ..SearchConfig::default()
                },
                &mut rand::thread_rng(),
            )
            .map_err(|error| error.to_string())?;
        Ok(result.best_move)
    }
}

impl Drop for UciSession {
    fn drop(&mut self) {
        self.evaluator.shutdown();
    }
}

fn legal_uci_move(board: &Board, uci: &str) -> Option<Move> {
    let mut selected = None;
    board.generate_moves(|moves| {
        for chess_move in moves {
            if standard_uci(board, chess_move) == uci {
                selected = Some(chess_move);
                return true;
            }
        }
        false
    });
    selected
}

fn repetitions(board: &Board, history: &[HistoryPosition]) -> u8 {
    history
        .iter()
        .filter(|entry| entry.board.same_position(board))
        .count()
        .saturating_add(1)
        .min(u8::MAX as usize) as u8
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut session = match UciSession::new() {
        Ok(session) => session,
        Err(error) => {
            let _ = writeln!(stdout, "info string initialization error: {error}");
            return;
        }
    };
    for line in stdin.lock().lines() {
        let Ok(command) = line else { break };
        match command.trim() {
            "uci" => {
                let _ = writeln!(stdout, "id name zero-rust-engine");
                let _ = writeln!(stdout, "id author ZERO-X");
                let _ = writeln!(stdout, "uciok");
            }
            "isready" => {
                let _ = writeln!(stdout, "readyok");
            }
            "ucinewgame" => session.new_game(),
            "quit" => break,
            _ if command.starts_with("position") => {
                if let Err(error) = session.set_position(&command) {
                    let _ = writeln!(stdout, "info string {error}");
                }
            }
            _ if command.starts_with("go") => match session.go(&command) {
                Ok(Some(chess_move)) => {
                    let _ = writeln!(
                        stdout,
                        "bestmove {}",
                        standard_uci(&session.board, chess_move)
                    );
                }
                Ok(None) => {
                    let _ = writeln!(stdout, "bestmove 0000");
                }
                Err(error) => {
                    let _ = writeln!(stdout, "info string {error}");
                    let _ = writeln!(stdout, "bestmove 0000");
                }
            },
            _ => {}
        }
        let _ = stdout.flush();
    }
}
