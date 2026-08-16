//! `zero-rust-engine`: high-throughput ZERO-X chess search primitives.

pub mod encoding;
pub mod evaluator;
pub mod mcts;
pub mod self_play;

pub use cozy_chess::{Board, Move};

#[cfg(feature = "python-extension")]
mod python {
    use std::sync::Arc;
    use std::time::Duration;

    use cozy_chess::{Board, Color, GameStatus, Move};
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::encoding::{
        encode_board_into, legal_policy_mask, standard_uci, HistoryPosition, INPUT_SIZE,
    };
    use crate::evaluator::SharedGpuEvaluator;
    use crate::mcts::{Mcts, SearchConfig};
    use crate::self_play::{generate_self_play, SelfPlayConfig};

    fn runtime_error(error: impl ToString) -> PyErr {
        PyRuntimeError::new_err(error.to_string())
    }

    #[pyclass]
    pub struct FastRustBoard {
        board: Board,
        /// Newest-first, matching the Python ZERO-X history convention.
        history: Vec<HistoryPosition>,
    }

    #[pymethods]
    impl FastRustBoard {
        #[new]
        #[pyo3(signature = (fen = None))]
        fn new(fen: Option<&str>) -> PyResult<Self> {
            let board = match fen {
                Some(fen) => fen
                    .parse::<Board>()
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
                None => Board::default(),
            };
            Ok(Self {
                board,
                history: Vec::with_capacity(512),
            })
        }

        fn fen(&self) -> String {
            self.board.to_string()
        }

        fn side_to_move(&self) -> &'static str {
            if self.board.side_to_move() == Color::White {
                "white"
            } else {
                "black"
            }
        }

        fn legal_moves(&self) -> Vec<String> {
            let mut legal = Vec::with_capacity(218);
            self.board.generate_moves(|moves| {
                for chess_move in moves {
                    legal.push(standard_uci(&self.board, chess_move));
                }
                false
            });
            legal
        }

        fn encode(&self) -> Vec<f32> {
            let repetitions = self
                .history
                .iter()
                .filter(|entry| entry.board.same_position(&self.board))
                .count()
                .saturating_add(1)
                .min(u8::MAX as usize) as u8;
            let mut encoded = [0.0; INPUT_SIZE];
            encode_board_into(&self.board, &self.history, repetitions, &mut encoded);
            encoded.to_vec()
        }

        fn policy_indices(&self) -> Vec<usize> {
            let mask = legal_policy_mask(&self.board);
            (0..crate::encoding::POLICY_SIZE)
                .filter(|index| mask.contains(*index))
                .collect()
        }

        /// Apply a legal UCI move. Parsing is deliberately validated against
        /// cozy-chess's generated legal list, so illegal moves cannot corrupt
        /// the board or its history planes.
        fn push_uci(&mut self, uci: &str) -> PyResult<()> {
            let mut selected: Option<Move> = None;
            self.board.generate_moves(|moves| {
                for chess_move in moves {
                    if standard_uci(&self.board, chess_move) == uci {
                        selected = Some(chess_move);
                        return true;
                    }
                }
                false
            });
            let chess_move = selected
                .ok_or_else(|| PyValueError::new_err("illegal UCI move for this position"))?;
            let repetitions = self
                .history
                .iter()
                .filter(|entry| entry.board.same_position(&self.board))
                .count()
                .saturating_add(1)
                .min(u8::MAX as usize) as u8;
            self.history.insert(
                0,
                HistoryPosition {
                    board: self.board.clone(),
                    repetitions,
                },
            );
            self.board.play(chess_move);
            Ok(())
        }

        fn reset(&mut self) {
            self.board = Board::default();
            self.history.clear();
        }

        /// A no-model diagnostic search that exercises the complete native
        /// move generation/MCTS path. Production analysis should use
        /// `generate_self_play_batch_rust` with a TorchScript model.
        #[pyo3(signature = (simulations = 128))]
        fn analyze_uniform(&self, py: Python<'_>, simulations: u32) -> PyResult<Py<PyDict>> {
            let board = self.board.clone();
            let display_board = board.clone();
            let history = self.history.clone();
            let repetitions = history
                .iter()
                .filter(|entry| entry.board.same_position(&board))
                .count()
                .saturating_add(1)
                .min(u8::MAX as usize) as u8;
            let result = py
                .detach(move || {
                    let evaluator = SharedGpuEvaluator::uniform(64, 64, Duration::from_micros(500));
                    let result =
                        Mcts::new(Arc::clone(&evaluator), 100_000, 64).and_then(|mut mcts| {
                            mcts.search(
                                &board,
                                &history,
                                repetitions,
                                SearchConfig {
                                    simulations,
                                    batch_size: 64,
                                    temperature: 0.0,
                                    add_root_noise: false,
                                    ..SearchConfig::default()
                                },
                                &mut rand::thread_rng(),
                            )
                        });
                    evaluator.shutdown();
                    result
                })
                .map_err(runtime_error)?;
            let result_dict = PyDict::new(py);
            result_dict.set_item(
                "best_move",
                result
                    .best_move
                    .map(|chess_move| standard_uci(&display_board, chess_move)),
            )?;
            result_dict.set_item("value", result.root_value)?;
            let visits = PyList::empty(py);
            for (chess_move, count) in result.visits {
                visits.append((standard_uci(&display_board, chess_move), count))?;
            }
            result_dict.set_item("visits", visits)?;
            Ok(result_dict.unbind())
        }
    }

    #[cfg(feature = "libtorch")]
    fn create_model_evaluator(
        model_path: &str,
        device: &str,
        slot_count: usize,
        batch_size: usize,
    ) -> PyResult<Arc<SharedGpuEvaluator>> {
        let device = match device {
            "cpu" => tch::Device::Cpu,
            "cuda" => tch::Device::Cuda(0),
            value if value.strip_prefix("cuda:").is_some() => {
                let index = value
                    .strip_prefix("cuda:")
                    .and_then(|text| text.parse::<usize>().ok())
                    .ok_or_else(|| PyValueError::new_err("device must be cpu, cuda, or cuda:N"))?;
                tch::Device::Cuda(index)
            }
            _ => return Err(PyValueError::new_err("device must be cpu, cuda, or cuda:N")),
        };
        SharedGpuEvaluator::torchscript(
            model_path,
            device,
            slot_count,
            batch_size,
            Duration::from_micros(200),
        )
        .map_err(runtime_error)
    }

    #[cfg(feature = "libtorch")]
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (candidate_path, incumbent_path, games, simulations = 64, device = "cpu", seed = 0, max_plies = 512, opening_random_plies = 4))]
    fn evaluate_torchscript_match(
        py: Python<'_>,
        candidate_path: &str,
        incumbent_path: &str,
        games: usize,
        simulations: u32,
        device: &str,
        seed: u64,
        max_plies: usize,
        opening_random_plies: usize,
    ) -> PyResult<Py<PyDict>> {
        if games == 0 || !games.is_multiple_of(2) {
            return Err(PyValueError::new_err(
                "games must be a positive even number",
            ));
        }
        if simulations == 0 || max_plies == 0 || opening_random_plies > max_plies {
            return Err(PyValueError::new_err("invalid native match configuration"));
        }
        if max_plies > crate::mcts::MAX_SEARCH_DEPTH {
            return Err(PyValueError::new_err(
                "max_plies exceeds native search depth",
            ));
        }
        let candidate_path = candidate_path.to_owned();
        let incumbent_path = incumbent_path.to_owned();
        let device = device.to_owned();
        let totals = py
            .detach(move || {
                let candidate = create_model_evaluator(&candidate_path, &device, 2, 1)
                    .map_err(|error| error.to_string())?;
                let incumbent = create_model_evaluator(&incumbent_path, &device, 2, 1)
                    .map_err(|error| error.to_string())?;
                let mut wins_a = 0_u32;
                let mut wins_b = 0_u32;
                let mut draws = 0_u32;

                for game_index in 0..games {
                    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(game_index as u64));
                    let mut board = Board::default();
                    let mut history = Vec::<HistoryPosition>::new();
                    for _ in 0..opening_random_plies {
                        let mut legal = Vec::with_capacity(218);
                        board.generate_moves(|moves| {
                            legal.extend(moves);
                            false
                        });
                        if legal.is_empty() {
                            break;
                        }
                        let chess_move = legal[rng.gen_range(0..legal.len())];
                        history.insert(
                            0,
                            HistoryPosition {
                                board: board.clone(),
                                repetitions: 1,
                            },
                        );
                        board.play(chess_move);
                    }

                    let mut mcts_a = Mcts::new(Arc::clone(&candidate), 100_000, 1)
                        .map_err(|error| error.to_string())?;
                    let mut mcts_b = Mcts::new(Arc::clone(&incumbent), 100_000, 1)
                        .map_err(|error| error.to_string())?;
                    let a_is_white = game_index < games / 2;
                    let mut repetitions = repetition_count(&board, &history);
                    for _ in 0..max_plies.saturating_sub(opening_random_plies) {
                        let status = board.status();
                        if matches!(status, GameStatus::Won | GameStatus::Drawn)
                            || repetitions >= 3
                            || crate::mcts::is_dead_position(&board)
                            || board.halfmove_clock() >= 100
                        {
                            break;
                        }
                        let config = SearchConfig {
                            simulations,
                            batch_size: 1,
                            temperature: 0.0,
                            add_root_noise: false,
                            ..SearchConfig::default()
                        };
                        let a_to_move = (board.side_to_move() == Color::White) == a_is_white;
                        let result = if a_to_move {
                            mcts_a.search(&board, &history, repetitions, config, &mut rng)
                        } else {
                            mcts_b.search(&board, &history, repetitions, config, &mut rng)
                        }
                        .map_err(|error| error.to_string())?;
                        let Some(chess_move) = result.best_move else {
                            break;
                        };
                        history.insert(
                            0,
                            HistoryPosition {
                                board: board.clone(),
                                repetitions,
                            },
                        );
                        board.play(chess_move);
                        repetitions = repetition_count(&board, &history);
                        mcts_a
                            .advance_to_with_context(chess_move, &board, &history, repetitions)
                            .map_err(|error| error.to_string())?;
                        mcts_b
                            .advance_to_with_context(chess_move, &board, &history, repetitions)
                            .map_err(|error| error.to_string())?;
                    }

                    match board.status() {
                        GameStatus::Won if board.side_to_move() == Color::White => wins_b += 1,
                        GameStatus::Won => wins_a += 1,
                        GameStatus::Drawn | GameStatus::Ongoing => draws += 1,
                    }
                }
                candidate.shutdown();
                incumbent.shutdown();
                Ok::<(u32, u32, u32), String>((wins_a, wins_b, draws))
            })
            .map_err(runtime_error)?;
        let output = PyDict::new(py);
        output.set_item("wins_a", totals.0)?;
        output.set_item("wins_b", totals.1)?;
        output.set_item("draws", totals.2)?;
        output.set_item("games", games)?;
        Ok(output.unbind())
    }

    fn repetition_count(board: &Board, history: &[HistoryPosition]) -> u8 {
        history
            .iter()
            .filter(|entry| entry.board.same_position(board))
            .count()
            .saturating_add(1)
            .min(u8::MAX as usize) as u8
    }

    #[cfg(not(feature = "libtorch"))]
    fn create_model_evaluator(
        _model_path: &str,
        _device: &str,
        _slot_count: usize,
        _batch_size: usize,
    ) -> PyResult<Arc<SharedGpuEvaluator>> {
        Err(PyRuntimeError::new_err(
            "zero_rust_engine was built without LibTorch; rebuild with --features libtorch,python-extension",
        ))
    }

    /// Process-lifetime cache for the TorchScript model. The module is loaded
    /// from disk exactly once and reused across every self-play batch; it is
    /// reloaded only when ``latest_model.ts`` has actually been updated by the
    /// Python training process (detected via the file modification timestamp).
    #[cfg(feature = "libtorch")]
    #[derive(Clone, PartialEq, Eq)]
    struct EvaluatorCacheKey {
        path: std::path::PathBuf,
        modified: std::time::SystemTime,
        size: u64,
        deployment_hash: Option<String>,
        device: String,
        slot_count: usize,
        batch_size: usize,
    }

    #[cfg(feature = "libtorch")]
    struct CachedEvaluator {
        key: EvaluatorCacheKey,
        evaluator: Arc<SharedGpuEvaluator>,
    }

    #[cfg(feature = "libtorch")]
    static EVALUATOR_CACHE: std::sync::Mutex<Option<CachedEvaluator>> = std::sync::Mutex::new(None);

    #[cfg(feature = "libtorch")]
    fn cached_model_evaluator(
        model_path: &str,
        device: &str,
        slot_count: usize,
        batch_size: usize,
    ) -> PyResult<Arc<SharedGpuEvaluator>> {
        let path = std::path::Path::new(model_path)
            .canonicalize()
            .map_err(runtime_error)?;
        let metadata = std::fs::metadata(&path).map_err(runtime_error)?;
        let hash_path = std::path::PathBuf::from(format!("{}.sha256", path.display()));
        let key = EvaluatorCacheKey {
            path,
            modified: metadata.modified().map_err(runtime_error)?,
            size: metadata.len(),
            deployment_hash: std::fs::read_to_string(hash_path)
                .ok()
                .map(|value| value.trim().to_owned()),
            device: device.to_owned(),
            slot_count,
            batch_size,
        };
        let mut cache = EVALUATOR_CACHE
            .lock()
            .map_err(|_| PyRuntimeError::new_err("evaluator cache mutex is poisoned"))?;
        if let Some(cached) = cache.as_ref() {
            if cached.key == key && cached.evaluator.is_healthy() {
                return Ok(Arc::clone(&cached.evaluator));
            }
        }
        let evaluator = create_model_evaluator(model_path, device, slot_count, batch_size)?;
        cache.replace(CachedEvaluator {
            key,
            evaluator: Arc::clone(&evaluator),
        });
        Ok(evaluator)
    }

    #[cfg(not(feature = "libtorch"))]
    fn cached_model_evaluator(
        model_path: &str,
        device: &str,
        slot_count: usize,
        batch_size: usize,
    ) -> PyResult<Arc<SharedGpuEvaluator>> {
        create_model_evaluator(model_path, device, slot_count, batch_size)
    }

    /// Generate native self-play and return ordinary Python dictionaries ready
    /// for the existing prioritized replay bridge. `model_path` must point to
    /// the exported TorchScript deployment module, not a Python state dict.
    #[pyfunction]
    #[pyo3(signature = (model_path, num_games, simulations = 400, batch_size = 256, device = "cuda", seed = None))]
    fn generate_self_play_batch_rust(
        py: Python<'_>,
        model_path: &str,
        num_games: usize,
        simulations: u32,
        batch_size: usize,
        device: &str,
        seed: Option<u64>,
    ) -> PyResult<Py<PyDict>> {
        if num_games == 0 {
            return Err(PyValueError::new_err("num_games must be greater than zero"));
        }
        if batch_size == 0 || batch_size > crate::mcts::MAX_BATCH {
            return Err(PyValueError::new_err("batch_size must be in 1..=256"));
        }
        if simulations == 0 {
            return Err(PyValueError::new_err(
                "simulations must be greater than zero",
            ));
        }
        let workers = num_games.clamp(1, 24);
        let model_path = model_path.to_owned();
        let device = device.to_owned();
        let seed = seed.unwrap_or(0x5EED_5EED);
        // Native self-play can run for minutes.  Detach it from the GIL so
        // Python can continue packing replay batches and launching SGD while
        // model loading, the Rust Rayon workers, and the LibTorch actor execute.
        let games = py
            .detach(move || {
                let evaluator =
                    cached_model_evaluator(&model_path, &device, workers * batch_size, batch_size)
                        .map_err(|error| error.to_string())?;
                generate_self_play(
                    evaluator,
                    num_games,
                    SelfPlayConfig {
                        simulations,
                        search_batch_size: batch_size,
                        workers,
                        ..SelfPlayConfig::default()
                    },
                    seed,
                )
                .map_err(|error| error.to_string())
            })
            .map_err(runtime_error);
        // The evaluator is kept alive in EVALUATOR_CACHE for the next batch;
        // it is only shut down when a model with a newer mtime replaces it.
        let games = games?;

        let output = PyDict::new(py);
        let game_list = PyList::empty(py);
        for game in games {
            let game_dict = PyDict::new(py);
            game_dict.set_item("result", game.result)?;
            game_dict.set_item("moves", game.moves)?;
            game_dict.set_item("termination", game.termination)?;
            let experiences = PyList::empty(py);
            for experience in game.experiences {
                let item = PyDict::new(py);
                let (policy_indices, policy_values): (Vec<u16>, Vec<f32>) =
                    experience.policy.into_iter().unzip();
                item.set_item("fen", experience.fen)?;
                item.set_item("policy_indices", policy_indices)?;
                item.set_item("policy_values", policy_values)?;
                item.set_item("value", experience.value)?;
                item.set_item("wdl", experience.wdl)?;
                item.set_item("target_kind", experience.target_kind)?;
                item.set_item("q_mcts", experience.q_mcts)?;
                item.set_item("material", (experience.material.0, experience.material.1))?;
                item.set_item("moves_left", experience.moves_left)?;
                item.set_item("repetitions", experience.repetitions)?;
                if let Some(opponent_policy) = experience.opponent_policy {
                    let (indices, values): (Vec<u16>, Vec<f32>) =
                        opponent_policy.into_iter().unzip();
                    item.set_item("opponent_policy_indices", indices)?;
                    item.set_item("opponent_policy_values", values)?;
                }
                if let Some(opponent_legal_policy) = experience.opponent_legal_policy {
                    item.set_item("opponent_legal_indices", opponent_legal_policy)?;
                }
                item.set_item("history_fens", experience.history_fens)?;
                item.set_item("history_repetitions", experience.history_repetitions)?;
                experiences.append(item)?;
            }
            game_dict.set_item("experiences", experiences)?;
            game_list.append(game_dict)?;
        }
        output.set_item("games", game_list)?;
        Ok(output.unbind())
    }

    #[pymodule]
    fn zero_rust_engine(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<FastRustBoard>()?;
        module.add_function(wrap_pyfunction!(generate_self_play_batch_rust, module)?)?;
        #[cfg(feature = "libtorch")]
        module.add_function(wrap_pyfunction!(evaluate_torchscript_match, module)?)?;
        Ok(())
    }
}
