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

    use cozy_chess::{Board, Color, Move};
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};

    use crate::encoding::{encode_board_into, legal_policy_mask, HistoryPosition, INPUT_SIZE};
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
                    legal.push(chess_move.to_string());
                }
                false
            });
            legal
        }

        fn encode(&self) -> Vec<f32> {
            let repetitions = 1 + self
                .history
                .iter()
                .filter(|entry| entry.board.same_position(&self.board))
                .count() as u8;
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
                    if chess_move.to_string() == uci {
                        selected = Some(chess_move);
                        return true;
                    }
                }
                false
            });
            let chess_move = selected
                .ok_or_else(|| PyValueError::new_err("illegal UCI move for this position"))?;
            let repetitions = 1 + self
                .history
                .iter()
                .filter(|entry| entry.board.same_position(&self.board))
                .count() as u8;
            if self.history.len() == 512 {
                let _ = self.history.pop();
            }
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
            let history = self.history.clone();
            let repetitions = 1 + history
                .iter()
                .filter(|entry| entry.board.same_position(&board))
                .count() as u8;
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
                result.best_move.map(|chess_move| chess_move.to_string()),
            )?;
            result_dict.set_item("value", result.root_value)?;
            let visits = PyList::empty(py);
            for (chess_move, count) in result.visits {
                visits.append((chess_move.to_string(), count))?;
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
        let device = if device.starts_with("cuda") {
            let index = device
                .split(':')
                .nth(1)
                .and_then(|text| text.parse::<usize>().ok())
                .unwrap_or(0);
            tch::Device::Cuda(index)
        } else {
            tch::Device::Cpu
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
        let key = EvaluatorCacheKey {
            path,
            modified: metadata.modified().map_err(runtime_error)?,
            size: metadata.len(),
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
        Ok(())
    }
}
