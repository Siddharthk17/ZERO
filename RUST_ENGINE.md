# `zero-rust-engine`

This document describes the Rust core of ZERO-X: its feature flags, build
requirements, board/policy ABI, evaluator transport, MCTS implementation,
self-play payload, PyO3 surface, and standalone UCI binary.

The Rust crate has two deliberately different jobs:

1. The production path loads a Python-exported TorchScript model and generates
   self-play games or candidate-gating matches.
2. The standalone `zero-rust-engine` executable uses a uniform evaluator. It is
   a legal chess/search/protocol smoke binary, not a model-backed playing
   engine.

The production training lifecycle, Python model, replay buffer, and frontend
are documented in [README.md](README.md).

## Crate layout and feature matrix

| File | Responsibility |
| --- | --- |
| `src/encoding.rs` | 121-plane input representation, legal mask, 4672-entry policy mapping, standard UCI conversion |
| `src/evaluator.rs` | Fixed-slot transport, batching actor, uniform backend, optional TorchScript backend |
| `src/mcts.rs` | Flat node arena, batched PUCT, virtual loss, draw handling, tree compaction |
| `src/self_play.rs` | Rayon game workers and replay-compatible target construction |
| `src/lib.rs` | PyO3 `FastRustBoard`, native self-play, native TorchScript matches, evaluator cache |
| `src/main.rs` | Standalone uniform-evaluator UCI session |

`Cargo.toml` defines two optional features:

| Feature | Enables |
| --- | --- |
| no feature | Core Rust library, tests, and uniform standalone UCI binary |
| `python-extension` | PyO3 module export |
| `libtorch` | `tch` TorchScript backend and native model evaluation |
| `libtorch,python-extension` | Production Python extension and model-backed native path |

The default feature set is empty so `cargo test --lib` can exercise the core
without linking Python, CUDA, or LibTorch.

## Rust requirements

- Rust `1.88` or newer.
- `cozy-chess 0.3.4` for legal move generation.
- `tch 0.24.0` when building with `libtorch`.
- PyO3 `0.29.1` when building with `python-extension`.
- A LibTorch/PyTorch runtime compatible with the locked `tch` binding for the
  model-backed path.

The direct dependencies are `arrayvec`, `cozy-chess`,
`crossbeam-channel`, `rand`, and `rayon`. Release builds use one codegen unit,
fat LTO, unwindable panics, and stripped symbols.

### LibTorch and PyTorch version warning

The Python project declares PyTorch `<2.13` beginning at `2.12.0.dev0`. The
locked `tch 0.24.0` binding targets LibTorch `2.11.0`. When the build uses the
Python PyTorch installation, the repository's build scripts require:

```text
ZERO_ALLOW_UNSUPPORTED_LIBTORCH=1
```

That variable causes the build scripts to set
`LIBTORCH_BYPASS_VERSION_CHECK=1`. It intentionally disables the version check;
it does not establish that the C++/CUDA ABI is safe. Run
`python scripts/preflight.py --device cpu` after every clean native build and
run the full-model or CUDA preflight before deployment.

## Build

All commands below run from the repository root.

### Core tests without LibTorch

```bash
cargo test --lib
```

This is the fastest way to validate the Rust-only encoding, evaluator state
machine, MCTS, draw handling, virtual loss, and subtree compaction. It does not
build the PyO3 extension and cannot validate TorchScript or CUDA.

### Linux with an external LibTorch tree

Use this route when `/opt/libtorch` or another explicit LibTorch distribution is
known to match `tch 0.24.0`:

```bash
export LIBTORCH=/opt/libtorch
unset LIBTORCH_USE_PYTORCH
cargo build --release --locked --features libtorch,python-extension
cp target/release/libzero_rust_engine.so zero_chess/zero_rust_engine.so
python scripts/preflight.py --device cpu --games 1 --simulations 1
```

The extension can also be placed at the repository root as
`zero_rust_engine.so`; `zero_chess.rust_bridge` tries the top-level module and
then the package-local module. The package-local path is the normal source-tree
installation path.

### Linux against the Python PyTorch installation

```bash
export LIBTORCH_USE_PYTORCH=1
export ZERO_ALLOW_UNSUPPORTED_LIBTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
cargo build --release --locked --features libtorch,python-extension
cp target/release/libzero_rust_engine.so zero_chess/zero_rust_engine.so
python scripts/preflight.py --device cpu --games 1 --simulations 1
```

`scripts/install.sh` automates this flow, including Python installation,
artifact placement, preflight, and pytest:

```bash
export LIBTORCH_USE_PYTORCH=1
export ZERO_ALLOW_UNSUPPORTED_LIBTORCH=1
bash scripts/install.sh
```

Do not interpret a successful Cargo build as runtime validation when the
unchecked override is in use.

### Windows

After installing the Python package, run:

```powershell
$env:LIBTORCH_USE_PYTORCH = "1"
$env:ZERO_ALLOW_UNSUPPORTED_LIBTORCH = "1"
powershell -ExecutionPolicy Bypass -File scripts/build_native.ps1
```

The script builds with `libtorch,python-extension`, finds the generated DLL,
copies it to `zero_chess/zero_rust_engine.pyd`, imports it, and runs the CPU
preflight unless `ZERO_SKIP_PREFLIGHT=1`.

### Local compiler and CUDA tuning

The repository does not make an ignored, machine-local `.cargo/config.toml`
part of its source contract. A deployment may add local `target-cpu` or CUDA
architecture settings, but those settings must be documented and reproduced
on the machine that owns the benchmark. A fresh checkout does not receive
local `.cargo` configuration.

## Cross-language ABI

Python and Rust implement one fixed representation. Any change to the channel
order, orientation, policy planes, or tensor shapes is an ABI change and must
update both implementations and their parity tests.

### Board input

Rust's `EncodedBoard` is `[f32; 7744]`:

```text
7744 = 121 input channels * 64 board squares
```

The TorchScript backend reshapes a batch to:

```text
[batch, 121, 8, 8]
```

The side to move defines the perspective. White coordinates are unchanged.
Black coordinates are rotated 180 degrees, and the piece groups are written as
own pieces followed by opponent pieces. The first 112 planes are eight
position slices of 14 planes each: current position plus seven newest history
positions, with six own-piece planes, six opponent-piece planes, and two
repetition planes per slice.

The final nine planes are:

| Plane | Meaning |
| ---: | --- |
| `112` | Side to move is White |
| `113` | Own short/kingside castling right |
| `114` | Own long/queenside castling right |
| `115` | Opponent short/kingside castling right |
| `116` | Opponent long/queenside castling right |
| `117` | En-passant file, after perspective orientation |
| `118` | Clipped fullmove number divided by 512 |
| `119` | Clipped halfmove clock divided by 100 |
| `120` | Side to move is in check |

`HistoryPosition` stores both a `cozy_chess::Board` and the position's
repetition count. The caller supplies the current repetition count separately.
Rust limits neural history to seven previous positions, but repetition
termination and context tracking use the full game history held by the caller.

### Policy coordinates

```text
POLICY_PLANES = 73
POLICY_SIZE  = 4672
```

An index is:

```text
plane * 64 + oriented_from_square
```

The planes are:

| Plane range | Meaning |
| --- | --- |
| `0..55` | Eight queen-like directions, seven distances per direction |
| `56..63` | Eight knight directions |
| `64..72` | Knight, bishop, and rook underpromotions, three pawn directions each |

Queen promotion is represented by the queen-like geometry. Underpromotion
geometry is validated before an index is returned. Legal masks use 73 `u64`
words because `4672 / 64 = 73`.

`cozy-chess` represents castling internally as a king move to the rook square.
`standard_destination()` converts the four orthodox cases to standard UCI
coordinates and policy destinations:

```text
e1h1 -> e1g1
e1a1 -> e1c1
e8h8 -> e8g8
e8a8 -> e8c8
```

`standard_uci()` and `move_to_policy_index()` use those standard destinations
at the Python and GUI boundary.

### TorchScript inputs and outputs

`zero_chess.model.export_torchscript()` traces a deployment wrapper with this
signature:

```text
input x:          [N, 121, 8, 8]
input legal_mask: [N, 4672]
```

The input tensor is BF16 on CUDA and FP32 on CPU. The legal mask is FP32. The
wrapper returns:

```text
output policy_logits: [N, 4672]
output scalar_value:  [N, 1]
output wdl:           [N, 3]
```

The value is the expected scalar over 128 bins centered uniformly in `[-1, 1]`.
The WDL order is win, draw, loss from the side-to-move perspective.

The wrapper masks illegal entries before returning policy logits. The Rust
TorchScript backend masks the returned policy logits again, applies softmax in
FP32, and copies the legal probabilities, values, and WDL arrays back to host
memory. It rejects wrong tuple types, wrong shapes, and non-finite output.

The sidecar written beside a deployment module is a one-line SHA-256 of the
TorchScript file. Rust includes that value in the process-lifetime evaluator
cache key when the sidecar is available.

## Evaluator transport

`SharedGpuEvaluator` is a fixed-slot CPU-to-backend transport shared by all
Rayon self-play workers. It avoids an unbounded queue of full tensors and lets
the actor coalesce leaves into one forward batch.

### Slot state machine

Each slot contains an encoded board, compact legal mask, output evaluation, and
optional error. Its atomic state is one of:

| State | Meaning |
| --- | --- |
| `FREE` | Available to a submitter |
| `PENDING` | Input is populated and queued for the actor |
| `RUNNING` | Actor included the slot in the current backend call |
| `DONE` | Output is available to the waiting ticket |
| `ERROR` | Backend failure is available to the waiting ticket |
| `CANCELLED` | Waiting timed out; actor releases the slot when safe |

The public operations are:

- `try_submit()` returns `Busy` immediately when no slot is available;
- `submit()` retries busy slots for up to 30 seconds;
- `evaluate()` submits and waits for one result;
- `EvaluationTicket::wait()` waits up to the same 30-second operation limit;
- `EvaluationTicket::wait_timeout()` supplies a shorter caller deadline;
- `shutdown()` closes the actor and wakes or fails pending work;
- `is_healthy()` reports whether the actor is open and finished state is false.

The actor collects at most the configured batch size and waits only the
configured coalescing interval for more requests. Backend errors and backend
panics are converted into `EvaluationError::Backend` and delivered to waiting
tickets. Tests cover both preserved backend error text and panic conversion.

The default native model backend uses pinned CPU staging tensors for CUDA,
converts input to BF16 on CUDA, calls TorchScript under `tch::no_grad`, and
copies FP32 output to host buffers. CPU uses FP32 input and does not allocate
CUDA staging buffers.

Shutdown waits at most two seconds for the actor. Rust cannot forcibly cancel a
foreign LibTorch or CUDA call, so a stuck backend thread may outlive the bounded
shutdown wait rather than deadlocking the host process.

### Uniform backend

`UniformBackend` fills legal policy entries uniformly, returns scalar value
`0.0`, and returns WDL `[0.0, 1.0, 0.0]`. It is useful for core tests, the
standalone binary, and `FastRustBoard.analyze_uniform()`. It is not a model
evaluation and must not be used as a strength measurement.

## Rust MCTS

### Fixed limits and defaults

The implementation in `src/mcts.rs` uses a flat arena. Child nodes occupy a
contiguous range and carry their incoming move; no child hash map is needed.

| Constant or setting | Value |
| --- | ---: |
| `MAX_BATCH` | `256` leaves |
| `MAX_SEARCH_DEPTH` | `512` plies |
| Default simulations | `800` |
| Default search batch | `64` |
| Default `c_puct` | `1.25` |
| Default FPU reduction | `0.25` |
| Default Dirichlet alpha | `0.3` |
| Default Dirichlet epsilon | `0.25` |
| Default root noise | enabled |
| Default temperature | `1.0` |

The PyO3 self-play function overrides the simulation count and search batch
from its arguments while retaining the other `SearchConfig` defaults. Native
gating uses deterministic temperature `0.0` and disables root noise.

### Search sequence

For each search:

1. Validate configuration and establish a root context containing the board,
   counters, repetition count, and history signature.
2. Return an immediate terminal result for checkmate, draw status, dead
   material, claimable draw, or halfmove termination.
3. Expand the root through the shared evaluator if necessary.
4. Mix Dirichlet noise into root priors when requested and not already applied
   for that root.
5. Select paths using dynamic PUCT and FPU. Visited child values are negated
   into the parent perspective.
6. Apply one virtual visit to every selected path node, with value offset `-1`
   at the root and `+1` below it.
7. Encode up to the configured batch of leaves and submit them to the fixed-slot
   evaluator.
8. Remove virtual loss, expand evaluated leaves, and back up the value with one
   sign change per ply.
9. Select the move by visit count at temperature zero or by visit-weighted
   sampling at a positive temperature.

The MCTS value is always from the active node's side-to-move perspective. A
checkmated side-to-move has terminal value `-1`; draws have value `0`.

### Draw and material handling

`is_dead_position()` handles the material cases that `cozy-chess` leaves to the
caller. Pawns, rooks, and queens make the position live. King-only and
single-minor positions are dead. Bishop-only positions are dead when all
bishops occupy one square color; any knight makes the multi-minor position
non-dead under this implementation.

Search also stops for:

- `cozy-chess` checkmate or drawn status;
- current repetition count at least three;
- a claimable threefold repetition now or on the next legal move;
- a claimable fifty-move position;
- halfmove clock at least 100;
- dead material.

Production Rust self-play treats claimable draw conditions as terminal draws.
There is no Rust option to decline a claim. Python `Board.outcome()` has a
separate `claim_draws` argument because the Python rules/test surface is more
configurable.

### Tree reuse

`advance_to_with_context()` makes the selected child the new root and copies
the reachable subtree into a spare arena. The remapped arena preserves visits,
value sums, priors, WDL, and expansion state for the selected line while
discarding siblings. This prevents an ordinary 512-ply game from accumulating
every old sibling in the fixed node pool.

If the child is unavailable, the next board or history context is invalid, or
compaction fails, the arena is reset. A context includes position identity,
fullmove and halfmove counters, repetition information, and the supplied
history signatures because the neural input is history-dependent.

## Rust self-play

### Defaults

`SelfPlayConfig::default()` is:

| Field | Value |
| --- | ---: |
| `simulations` | `400` |
| `fast_simulations` | `35` |
| `full_search_probability` | `0.20` |
| `fast_search_weight` | `0.25` |
| `search_batch_size` | `64` |
| `max_plies` | `512` |
| `workers` | `24` |
| `node_capacity` | `250000` nodes per game |

`generate_self_play()` validates positive budgets, a full-search probability and
fast-search weight in `[0, 1]`, a search batch no larger than 256, and a maximum
game length no larger than 512. The PyO3 entry point chooses
`workers = num_games.clamp(1, 24)` and overrides the full-search simulation,
evaluator batch, and fast-search weight arguments.

### Per-ply behavior

For every game:

1. Start from the standard initial board.
2. Check terminal status, repetition/claimable draw, dead material, and the
   halfmove clock.
3. Sample whether this ply receives full search. Full search uses the configured
   `simulations`; the other path uses `fast_simulations`.
4. Search with root noise enabled. Use temperature `1.0` for plies `0..11` and
   `0.05` thereafter.
5. Record every pending training position. Full-search policy targets receive
   weight `1.0`; fast-search policy targets receive `fast_search_weight`.
6. Play the selected standard UCI move and advance the tree with full history
   context.

The resulting game can end as:

| `termination` | Target kind |
| --- | --- |
| `checkmate` | `terminal` |
| `draw` | `terminal` |
| `no_legal_move` | `terminal` draw-shaped result |
| `max_plies` | `truncated` |

An unfinished game at the fixed maximum is not relabelled as a decisive result.
Its experiences keep `q_mcts` and a draw-shaped WDL field for schema
completeness, while Python training masks their WDL loss.

### Experience construction

For every recorded position, Rust emits:

| Field | Meaning |
| --- | --- |
| `fen` | Current standard FEN |
| `policy_indices` | Sparse legal policy coordinates with non-zero visit counts |
| `policy_values` | Visit probabilities aligned with `policy_indices` |
| `policy_weight` | Confidence weight for the policy target |
| `value` | Final result from the position's side-to-move perspective |
| `wdl` | One-hot win/draw/loss from the same perspective |
| `target_kind` | `terminal` or `truncated` |
| `q_mcts` | Root MCTS value before terminal target construction |
| `material` | Normalized white and black material labels |
| `moves_left` | Remaining plies clipped at 100 and divided by 100 |
| `repetitions` | Current position occurrence count |
| `history_fens` | Up to seven prior FENs, newest first |
| `history_repetitions` | Counts aligned with `history_fens` |
| `opponent_policy_indices` | Following policy, when available |
| `opponent_policy_values` | Following probabilities |
| `opponent_legal_indices` | Legal policy coordinates for that following position |

The following opponent policy is present when the immediately following
position is recorded. It is not a reward and is not part of the canonical loss
because the conditioning move is not stored.

The game-level payload contains:

```text
result       +1.0 for a White win, -1.0 for a Black win, 0.0 for a draw
moves        standard UCI strings
termination  termination label
experiences  list of sparse replay dictionaries
```

The Python bridge validates WDL length, normalizes sparse policy, clamps target
ranges and policy confidence, preserves target kind and history, and inserts
`Experience` objects with initial priority `1.0`. Native integer policy indices
are bounds-checked without repeating legal move generation.

## PyO3 API

The module is named `zero_rust_engine`. `zero_chess.rust_bridge._engine()` first
tries that top-level module and then `zero_chess.zero_rust_engine`.

### `FastRustBoard`

`FastRustBoard(fen=None)` wraps a `cozy_chess::Board` and newest-first history.

| Method | Return or effect |
| --- | --- |
| `fen()` | Current standard FEN |
| `side_to_move()` | `"white"` or `"black"` |
| `legal_moves()` | Standard UCI strings |
| `encode()` | Flattened 7744-element `f32` input encoding |
| `policy_indices()` | Legal coordinates in the 4672-entry policy space |
| `push_uci(uci)` | Validates and applies one legal UCI move |
| `reset()` | Restores the starting position and clears history |
| `analyze_uniform(simulations=128)` | Native diagnostic search with no model |

`push_uci()` resolves the supplied string against the generated legal move list.
An invalid move raises a Python `ValueError` without changing board or history.
`analyze_uniform()` returns a dictionary with `best_move`, `value`, and `visits`.
It deliberately does not claim model strength.

### `encode_training_batch`

`encode_training_batch(fens, history_fens, repetitions, history_repetitions)`
bulk-encodes replay samples in Rust and returns writable byte buffers for the
7744-element `f32` inputs and dense `u8` legal-policy masks. The Python training
loop converts those buffers directly into tensors, avoiding one PyO3 call and
one Python FEN/move-generation pass per sample.

### `generate_self_play_batch_rust`

Signature and defaults:

```python
generate_self_play_batch_rust(
    model_path,
    num_games,
    simulations=400,
    batch_size=256,
    device="cuda",
    seed=None,
    fast_search_weight=0.25,
)
```

Validation rules:

- `num_games > 0`;
- `simulations > 0`;
- `1 <= batch_size <= 256`;
- `device` is `cpu`, `cuda`, or `cuda:N`.
- `fast_search_weight` is finite and in `[0, 1]`.

The function releases the Python GIL while the model is loaded/cached and Rust
generates games. It returns ordinary Python dictionaries under the `games` key.
The `model_path` must be an exported TorchScript deployment module, not a
PyTorch `.pt` checkpoint payload.

### `evaluate_torchscript_match`

This function is compiled only with `libtorch`:

```python
evaluate_torchscript_match(
    candidate_path,
    incumbent_path,
    games,
    simulations=64,
    device="cpu",
    seed=0,
    max_plies=512,
    opening_random_plies=4,
    batch_size=8,
    workers=0,
)
```

`games` must be positive and even. `max_plies` must be positive and no larger
than 512; opening plies cannot exceed the maximum. Candidate A receives White
in the first half and Black in the second half. Matches use temperature zero,
no root noise, parallel native games, and batched evaluator leaves. `workers=0`
automatically selects up to 24 workers. The result dictionary contains
`wins_a`, `wins_b`, `draws`, and `games`; wins are always reported from
candidate A's perspective, independent of color.

The Python `arena.gate_checkpoints()` function exports a temporary TorchScript
module for the candidate and reuses the incumbent deployment module when one
is supplied by the master. It calls this native function by default and
computes the score interval used by the master. It falls back to Python MCTS
only when `ZERO_GATE_NATIVE=0`.

### Evaluator cache

For native self-play, the cache key contains:

- canonical model path;
- model modification time;
- model size;
- optional `<model>.sha256` contents;
- device;
- slot count;
- evaluator batch size.

The cache preserves the evaluator across self-play batches. A changed key
replaces the cached evaluator. If the old evaluator is executing foreign
backend code, shutdown uses the bounded two-second behavior described above.

## Python bridge example

The following is the smallest representative model-backed call. It assumes the
native extension was built and the accepted checkpoint exists:

```python
from pathlib import Path

from zero_chess.model import export_torchscript, load_model
from zero_rust_engine import generate_self_play_batch_rust

checkpoint = Path("checkpoints/zero_x/accepted.pt")
deployment = Path("checkpoints/zero_x/manual_model.ts")

model = load_model(checkpoint, device="cuda")
export_torchscript(deployment, model, device="cuda")

payload = generate_self_play_batch_rust(
    str(deployment),
    num_games=24,
    simulations=400,
    batch_size=256,
    device="cuda",
    seed=1234,
)
print(len(payload["games"]))
```

For CPU execution, export and call with `device="cpu"`; the native extension
must still have been built with `libtorch,python-extension`.

The canonical master does the same export automatically, but uses
`checkpoints/zero_x/latest_model.ts` as its deployment path and updates it only
when an accepted model changes.

## Standalone UCI binary

Build the default feature set:

```bash
cargo build --release --locked
```

Run it:

```bash
./target/release/zero-rust-engine
```

The binary creates:

- a uniform evaluator with 256 slots;
- a 250,000-node MCTS arena;
- a search batch size of 64;
- no TorchScript model loading.

It supports this subset of UCI:

| Command | Behavior |
| --- | --- |
| `uci` | Reports engine identity and `uciok` |
| `isready` | Reports `readyok` |
| `ucinewgame` | Resets board, history, and tree |
| `position startpos ...` | Sets a validated starting position and moves |
| `position fen ...` | Sets a validated six-field FEN and moves |
| `go nodes N` | Searches `N` simulations, default 800, clamped to 1-50000 |
| `quit` | Stops the session |

It does not load a checkpoint, expose UCI options, implement clock management,
support `go movetime`, implement `stop`, or use the Python model. A
`bestmove` from this executable is a uniform-prior MCTS smoke result.

## Testing and diagnostics

### Rust

```bash
cargo test --lib
```

The embedded Rust tests cover:

- standard castling UCI/policy coordinate conversion;
- uniform evaluator backend behavior;
- evaluator backend panic and error propagation;
- subtree compaction and reuse;
- full-history repetition identity;
- dead-material classification;
- WDL normalization and safe fallback;
- invalid search parameter rejection;
- virtual-loss perspective and cleanup;
- evaluator slot starvation without deadlock.

### Native Python preflight

```bash
python scripts/preflight.py --device cpu --games 1 --simulations 1
python scripts/preflight.py --device cuda --games 1 --simulations 1
python scripts/preflight.py --device cuda --full-model
```

The preflight is the repository's practical native smoke test. It imports the
extension, exports a tiny or full deployment module, executes native self-play,
ingests at least one experience, and runs one training step. It is not a
long-run stability, throughput, or strength benchmark.

### ABI parity

`tests/test_rust_bridge.py` compares a built `FastRustBoard` with the Python
board for starting-position encoding, legal policy indices, history encoding,
and castling. The test is skipped when the native extension is absent. Any
change to `src/encoding.rs` should be made together with the matching Python
change in `zero_chess/encoding.py` and an explicit parity test.

## Troubleshooting

### `zero_rust_engine` cannot be imported

Build with both production features and place the artifact where the bridge can
find it:

```bash
cargo build --release --locked --features libtorch,python-extension
cp target/release/libzero_rust_engine.so zero_chess/zero_rust_engine.so
python -c "import importlib; importlib.import_module('zero_chess.zero_rust_engine')"
```

When using Python PyTorch, confirm that the explicit ABI override was set and
then run the CPU preflight.

### TorchScript shape or tuple errors

The deployment module must accept `[N,121,8,8]` and `[N,4672]`, and return a
three-tensor tuple with shapes `[N,4672]`, `[N,1]`, and `[N,3]`. Export with
`zero_chess.model.export_torchscript()` rather than tracing an arbitrary model
class.

### CUDA inference is rejected

Check both CUDA availability and BF16 support:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
```

Use `--device cpu` for a CPU smoke test. CPU does not remove the native
extension requirement for production self-play.

### Native search stalls or reports busy/timeout

The evaluator has a 30-second submit/wait bound and a fixed slot pool. Reduce
the requested evaluator batch or simulation budget while diagnosing the
problem. Inspect the worker's consecutive-error message and run a one-game,
one-simulation preflight before restarting a long run.

### The standalone binary looks weak

That is expected. `src/main.rs` uses `UniformBackend` by design. Use the Python
UCI command with an accepted checkpoint for model-backed interactive play, or
use `generate_self_play_batch_rust` for production native self-play.
