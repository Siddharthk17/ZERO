# ZERO-X

ZERO-X is a tabula-rasa chess research engine. It generates training data by
self-play, trains a neural network from that data, and promotes new models only
after they pass a candidate-versus-incumbent gate. The production self-play and
search implementation is Rust. Python owns model construction, optimization,
checkpoint management, replay ingestion, gating orchestration, UCI, and the
WebSocket service. A Next.js application provides a browser interface to that
service.

This repository is an engineering and research system, not a claim of playing
strength. Elo, positions per second, GPU utilization, tactical performance,
and convergence must be measured on the deployment machine and recorded with
the experiment that produced them.

## Contents

- [Scope and learning objective](#scope-and-learning-objective)
- [Architecture](#architecture)
- [Repository map](#repository-map)
- [Requirements](#requirements)
- [Installation](#installation)
- [Validation](#validation)
- [Training](#training)
- [Artifacts and state](#artifacts-and-state)
- [Model and training contract](#model-and-training-contract)
- [Native self-play contract](#native-self-play-contract)
- [Interactive engine and API](#interactive-engine-and-api)
- [Frontend](#frontend)
- [Lichess integration](#lichess-integration)
- [Tests](#tests)
- [Operational limitations](#operational-limitations)
- [License](#license)

## Scope and learning objective

The default experiment starts from a newly initialized `ZeroNet` unless a
checkpoint is explicitly resumed. The training loop does not consume human
game databases, opening books, Polyglot books, tablebases, or pretrained model
weights. The chess rules engine and the policy-coordinate mapping are built-in
because they are required to generate legal games and represent moves.

The engine uses a strict zero-sum terminal objective:

| Game result | White value | Black value | WDL target from the winning side's perspective |
| --- | ---: | ---: | --- |
| `1-0` | `+1` | `-1` | win / loss |
| `0-1` | `-1` | `+1` | loss / win |
| `1/2-1/2` | `0` | `0` | draw |

MCTS backs up a value with one sign change per ply. Python's legacy search
helpers can apply a small, bounded contempt offset for move selection and UCI
reporting; that offset is not stored in replay and is not a training target.
The Rust production self-play path does not use capture, mobility,
aggression, king-safety, or early model-value reward shaping.

There is one important qualification to the phrase "tabula rasa": Rust emits a
normalized material label as an auxiliary supervised target. It uses the
orthodox values pawn `1`, knight `3`, bishop `3`, rook `5`, and queen `9`, then
divides each side's total by `39`. This label is not a reward, is not consulted
by MCTS, and does not determine the game result.

## Architecture

### End-to-end loop

The canonical path in `train_master.py` is:

1. Load a valid model payload, preferring `--resume`, then `accepted.pt`, then
   `latest.pt`, then the newest managed `zero_iter_*.pt` checkpoint.
2. Create `accepted.pt` when no valid checkpoint exists.
3. Export the accepted model to `latest_model.ts` and write its SHA-256 sidecar.
4. Run native Rust self-play in a background worker. Rust loads the TorchScript
   module, runs Rayon workers and flat-arena MCTS, and evaluates leaves through
   a shared fixed-slot evaluator actor.
5. Convert the returned sparse policy/value/WDL dictionaries into validated
   `Experience` objects and insert them into the Python prioritized replay
   buffer.
6. Train the Python `ZeroNet` while enforcing the configured replay-to-generated
   sample ratio.
7. Save a candidate at the configured interval and gate it against the last
   accepted checkpoint.
8. Replace `accepted.pt` and `latest_model.ts` only when the candidate passes
   the gate. A rejected candidate restores the accepted model and optimizer.

The native evaluator is cached for the process lifetime. It is reused while the
deployment module's canonical path, modification time, size, sidecar hash,
device, slot count, and batch size remain unchanged. A changed deployment
module causes the cache to be replaced for the next native call.

### Ownership boundary

| Responsibility | Owner | Canonical implementation |
| --- | --- | --- |
| Legal move generation for production self-play | Rust | `cozy-chess`, `src/self_play.rs` |
| Production MCTS | Rust | `src/mcts.rs` |
| Batched TorchScript evaluation | Rust | `src/evaluator.rs` |
| Self-play orchestration | Rust, called through PyO3 | `src/self_play.rs`, `src/lib.rs` |
| Replay validation and persistence | Python | `zero_chess/replay.py`, `zero_chess/rust_bridge.py` |
| Neural-network definition and optimization | Python | `zero_chess/model.py`, `zero_chess/training.py` |
| Checkpoints and run state | Python | `zero_chess/checkpoint.py`, `train_master.py` |
| Candidate gating | Python orchestration, Rust native matches by default | `zero_chess/arena.py`, `src/lib.rs` |
| UCI engine used by `zero-uci` and the WebSocket server | Python | `zero_chess/uci.py` |
| Browser client | TypeScript/Next.js | `frontend/` |

The Python board and Python MCTS are not the canonical self-play generator.
They remain important for deterministic rules tests, Python UCI play, the
explicit non-native gate used by tests, and isolated evaluation. Candidate
gating itself is orchestrated through the Python `arena` module; its default
match implementation is the native Rust TorchScript match.

### Search and evaluation topology

Each Rust self-play game owns one MCTS tree and one flat node arena. Rayon
workers run games in parallel. Leaves from those trees are copied into fixed
CPU slots and coalesced by one evaluator actor before a TorchScript forward
pass. Search trees never share nodes across games. The evaluator is shared;
the trees are not.

The Rust arena uses contiguous node storage and subtree compaction when a
played child becomes the next root. The normal path does not allocate a boxed
node or a per-node child map. Virtual loss makes paths selected by concurrent
leaf collection temporarily less attractive, and backup alternates the value
perspective exactly.

## Repository map

```text
README.md                         Project guide and operating contract
RUST_ENGINE.md                    Rust core, native build, and ABI reference
Cargo.toml                        Rust crate and feature definitions
pyproject.toml                    Python package, CLI entry points, and test config
setup.py                          Optional PyO3 extension build hook

src/
  encoding.rs                     Rust/Python-compatible 121-plane and 4672-policy ABI
  evaluator.rs                    Fixed-slot evaluator transport and TorchScript backend
  mcts.rs                         Flat-arena batched PUCT search
  self_play.rs                    Parallel games and replay target construction
  lib.rs                          PyO3 module and native match/self-play entry points
  main.rs                         Standalone uniform-evaluator UCI binary

zero_chess/
  board.py                        Complete Python chess rules engine
  move.py                         Immutable move representation and UCI conversion
  constants.py                    Board, piece, castling, and coordinate constants
  zobrist.py                      Deterministic position hashing
  encoding.py                     Python side of the neural input/policy ABI
  model.py                        ZeroNet, checkpoints, hashes, and TorchScript export
  training.py                     Replay encoding, losses, optimizer, and scheduler
  replay.py                       Thread-safe prioritized replay and pickle snapshots
  checkpoint.py                   Atomic checkpoint index and pruning
  rust_bridge.py                  Native payload validation and history provenance
  arena.py                        Candidate gating and confidence interval calculation
  self_play.py                    Thin Rust self-play command-line facade
  uci.py                          Python UCI engine using Python MCTS
  websocket_server.py             FastAPI/WebSocket bridge to Python UCI

train_master.py                   Canonical long-running training process

scripts/
  install.sh                      Linux install, native build, preflight, and pytest
  build_native.ps1                Windows native build and preflight
  preflight.py                    Native export -> self-play -> replay -> train smoke test
  perft.py                        Python rules-engine perft runner
  train_loop.sh                   Unix restart wrapper
  train_loop.ps1                  PowerShell restart wrapper

frontend/                         Separate Next.js browser application
tests/                            Python regression and optional integration tests
```

Generated state is intentionally ignored by Git. The usual ignored paths are
`checkpoints/`, `data/`, `logs/`, `target/`, Python caches, and the frontend's
`node_modules/` and `.next/` directories.

## Requirements

### Python and Rust

- Python `>=3.11,<3.15`.
- Rust `1.88` or newer, as declared by `Cargo.toml`.
- A Python environment with the declared NumPy, PyTorch, FastAPI, Uvicorn,
  and WebSockets dependencies.
- The development extra adds `pytest` and `ruff`.

The pure Rust core and its unit tests do not require LibTorch. Production
self-play, native gating, and the canonical master do require the PyO3
extension built with both `libtorch` and `python-extension` features. The
master requires that extension even when `--device cpu` is selected.

### CUDA

CUDA training and CUDA TorchScript evaluation require a GPU for which
`torch.cuda.is_bf16_supported()` is true. The Python model deliberately
rejects CUDA inference/export without native BF16 support. CPU inference and
CPU training are supported, but CPU production training still needs the native
extension because self-play remains Rust-owned.

### Native ABI warning

The repository declares PyTorch `<2.13` beginning at `2.12.0.dev0`, while the
locked `tch 0.24.0` binding targets LibTorch `2.11.0`. When building against a
Python PyTorch installation, the version check must be explicitly bypassed
with `ZERO_ALLOW_UNSUPPORTED_LIBTORCH=1`. That flag disables a version guard; it
does not prove ABI compatibility. Run the native preflight after every clean
build and treat a successful compile alone as insufficient validation.

### Frontend

Node.js and npm are required only for `frontend/`. The checked-in
`frontend/package-lock.json` is the dependency lockfile. The frontend is a
separate Next.js process and does not proxy the Python service.

## Installation

Commands below assume the repository root as the working directory.

### Python-only development install

This installs the Python package and development tools. It is sufficient for
the rules engine, Python tests, Python UCI with its uniform fallback, and
non-native development. It is not sufficient for production self-play.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Linux native install against Python PyTorch

Install the Python dependencies first, then build the extension and run the
repository's native preflight and tests:

```bash
export LIBTORCH_USE_PYTORCH=1
export ZERO_ALLOW_UNSUPPORTED_LIBTORCH=1
bash scripts/install.sh
```

`scripts/install.sh` performs the following operations:

1. Installs the editable package with development dependencies.
2. Builds `libzero_rust_engine` with `libtorch,python-extension`.
3. Copies the extension into `zero_chess/zero_rust_engine.so`.
4. Imports the extension.
5. Runs a one-game, one-simulation CPU preflight unless
   `ZERO_SKIP_PREFLIGHT=1`.
6. Runs `python -m pytest`.

The script defaults to building native code. `ZERO_BUILD_NATIVE=0` skips the
build step only; the script still imports `zero_chess.zero_rust_engine`, so it
is not a pure-Python installer unless an extension is already present. Use the
direct Python installation commands above for a pure-Python environment.

### Linux native build against an external LibTorch distribution

When using a separately installed LibTorch tree, build directly with Cargo:

```bash
export LIBTORCH=/opt/libtorch
unset LIBTORCH_USE_PYTORCH
cargo build --release --locked --features libtorch,python-extension
cp target/release/libzero_rust_engine.so zero_chess/zero_rust_engine.so
python scripts/preflight.py --device cpu --games 1 --simulations 1
```

The external LibTorch distribution must match the locked `tch` ABI. Do not set
the unchecked override unless the deployment ABI has been deliberately
validated.

### Windows native build

Install the Python package first, then set the explicit ABI override when using
the Python PyTorch installation:

```powershell
$env:LIBTORCH_USE_PYTORCH = "1"
$env:ZERO_ALLOW_UNSUPPORTED_LIBTORCH = "1"
```

The script copies the generated DLL to
`zero_chess/zero_rust_engine.pyd` and runs the CPU preflight unless
`ZERO_SKIP_PREFLIGHT=1`.

### Frontend install

```bash
cd frontend
npm ci
npm run lint
npm run build
```

Start the development server with:

```bash
npm run dev
```

The default browser endpoints are `ws://localhost:8765` for WebSocket engine
requests and `http://localhost:8765` for training history. Override them before
starting Next.js when the backend is elsewhere:

```bash
NEXT_PUBLIC_ZERO_WS_URL=ws://127.0.0.1:8765 \
NEXT_PUBLIC_ZERO_API_URL=http://127.0.0.1:8765 \
npm run dev
```

## Validation

Run the checks that match the layer being changed.

### Rust core

```bash
cargo test --lib
```

This uses the default feature set and does not link LibTorch. It covers the
Rust encoding parity assumptions, evaluator error propagation, MCTS subtree
reuse, draw detection, value backup, virtual loss, and fixed-slot starvation
cases.

### Python rules and training stack

```bash
python -m pytest
ruff check .
```

The native bridge test skips the optional extension-specific test when the
extension is not installed. A passing Python test run without the extension is
therefore not proof that native self-play works.

### Rules-engine perft

```bash
python scripts/perft.py --depth 5
```

For the standard starting position, depth 5 must be `4865609`. The script
prints every depth from 1 through the requested depth. A custom FEN can be
passed as the positional argument.

### Native end-to-end preflight

The preflight exports a model, runs native self-play, ingests the returned
experiences, and executes one Python training step:

```bash
python scripts/preflight.py --device cpu --games 1 --simulations 1
python scripts/preflight.py --device cuda --games 1 --simulations 1
python scripts/preflight.py --device cuda --full-model
```

The CUDA commands require a CUDA-capable, BF16-capable PyTorch installation.
The default preflight model is deliberately tiny. `--full-model` uses the
default 12-block, 256-channel model and is the relevant smoke test for the
production architecture.

## Training

### Direct launch

The canonical training process is:

```bash
python train_master.py --device cuda
```

A representative explicit launch is:

```bash
python train_master.py \
  --device cuda \
  --games-per-batch 200 \
  --simulations 400 \
  --eval-batch-size 256 \
  --training-batch-size 1024
```

For a new run that must not reuse existing accepted checkpoints, replay, or
run state:

```bash
python train_master.py --device cuda --fresh
```

`--fresh` refuses to start if the accepted checkpoint, `latest.pt`, the replay
path, or the run-state path already exists. It does not delete those files.

### Restart wrappers

Unix:

```bash
bash scripts/train_loop.sh
```

The Unix wrapper always starts the master with `--device cuda` and forwards
additional arguments. It adds `--resume checkpoints/zero_x/accepted.pt` when
that file exists, uses recoverable state when run state or replay exists, and
otherwise adds `--fresh`.

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_loop.ps1 -Device cuda -Days 31
```

The PowerShell wrapper exposes the common training parameters directly. Both
wrappers retry non-zero exits other than `0` and `130`; inspect the persisted
run state before using them for unattended operation because an expired run
deadline causes the master to exit non-zero.

### Master options

The defaults below are defined in `train_master.py` and its `TrainConfig`.

| Option | Default | Meaning |
| --- | ---: | --- |
| `--resume` | none | Explicit checkpoint payload to try first |
| `--device` | `cuda` when available, otherwise `cpu` | Model and native evaluator device |
| `--days` | `31` | Duration used only when creating a new run state |
| `--games-per-batch` | `128` | Native games returned per self-play call |
| `--simulations` | `400` | Full-search budget passed to Rust |
| `--eval-batch-size` | `256` | Native evaluator/MCTS leaf batch limit |
| `--training-batch-size` | `1024` | Python SGD batch size |
| `--channels` | `256` | New-model trunk channels |
| `--blocks` | `12` | New-model residual block count |
| `--policy-channels` | `64` | Policy head intermediate channels |
| `--target-replay-ratio` | `4.0` | Maximum training samples per generated position |
| `--warmup-experiences` | `10000` | Replay size required before SGD begins |
| `--replay-capacity` | `4000000` | Hot replay capacity |
| `--replay-path` | `checkpoints/zero_x/replay.pkl` | Replay snapshot path |
| `--history-path` | `data/training_games.jsonl` | Native game provenance JSONL |
| `--candidate-interval` | `5000` | Training steps between candidate gates |
| `--gate-games` | `40` | Even number of gate games |
| `--gate-simulations` | `64` | Search budget for each gate game |
| `--gate-device` | `cpu` | Device used for native gate matches |
| `--replay-save-interval` | `5000` | Training steps between replay snapshots |
| `--shutdown-timeout` | `300` seconds | Worker join timeout during shutdown |
| `--self-play-timeout` | `600` seconds | Maximum permitted stalled native call |
| `--run-state-path` | `checkpoints/zero_x/run_state.json` | Persisted deadline file |
| `--disable-gating` | off | Accept candidates immediately; use for smoke tests only |
| `--fresh` | off | Refuse existing accepted/latest/replay/run-state state |
| `--seed` | `0x5EED_5EED` | Python, Torch, and native seed base |

### Run lifecycle and acceptance

The master starts the self-play worker before replay warmup is complete. The
worker waits for `latest_model.ts`, calls Rust, requires exactly the requested
game count, ingests at least one experience, appends game history, and retries
transient failures. Five consecutive errors or a call exceeding
`--self-play-timeout` make the worker fatal.

Once replay reaches `--warmup-experiences`, the master trains only while the
number of samples trained remains below
`generated_positions * --target-replay-ratio`. This prevents optimization from
running arbitrarily far ahead of the data generator.

At each candidate interval, the master runs a color-balanced candidate-A versus
incumbent-B match. The default native gate uses the requested number of even
games, four random opening plies, no root noise, deterministic move selection,
and a 512-ply limit. It computes a score fraction, an Elo-style difference,
and a conservative Hoeffding interval for the mixed win/draw/loss score. The
candidate is accepted only when:

```text
score_low > 0.5
```

The default gate requires `evaluate_torchscript_match` from the native
extension. The Python MCTS fallback is available only when
`ZERO_GATE_NATIVE=0`; that setting is intended for tests and isolated
development, not as an accidental production fallback.

The run deadline is stored as Unix timestamps in `run_state.json`. A restart
with the same run-state file preserves the original deadline even when a new
`--days` value is supplied. When the deadline is reached, the master saves the
accepted model and replay in its finalization path and exits.

## Artifacts and state

The master uses `checkpoints/zero_x/` by default.

| Path | Format | Purpose |
| --- | --- | --- |
| `accepted.pt` | PyTorch payload dictionary | Last model accepted by the gate; training checkpoint, not TorchScript |
| `latest_model.ts` | TorchScript module | Deployment module consumed by Rust self-play |
| `latest_model.ts.sha256` | One-line SHA-256 text | Deployment provenance and native evaluator cache invalidation |
| `zero_iter_XXXXXXX.pt` | PyTorch payload dictionary | Managed accepted checkpoint at a training iteration |
| `latest.pt` | Copy of a managed checkpoint | Latest managed checkpoint |
| `index.json` | JSON list | Managed checkpoint metadata and pruning index |
| `candidate_XXXXXXX.pt` | PyTorch payload dictionary | Temporary candidate removed after gate decision |
| `replay.pkl` | Atomic pickle snapshot | Prioritized replay contents and sampling state |
| `run_state.json` | JSON | Run-state version, start timestamp, and deadline |

### Checkpoint payload

`save_model()` writes a dictionary containing:

- `architecture`: currently `zero_x_swiglu_seresnet_v3`;
- `config`: model dimensions and ABI settings;
- `model_hash`: SHA-256 over sorted state-dictionary names, dtypes, shapes,
  and tensor bytes;
- `model`: the PyTorch state dictionary;
- optional caller metadata such as `iteration`, `optimizer`, `metrics`, and
  `elo`.

Loading is strict. The loader recovers dimensions such as channel count, block
count, policy channels, policy size, and value-bin count from the state when
necessary, then validates `model_hash` if present. A `.pt` payload is not a
native executable model and must be exported before Rust can consume it.

### Replay snapshot

The hot replay buffer is RAM-resident while training. The master saves it to
`replay.pkl` periodically and during shutdown by default. The snapshot includes
experiences, priorities, ring cursor, slot generations, sampling RNG state,
capacity, and metadata. The current metadata includes:

```text
schema_version: 2
encoding_version: zero-x-121x73-v1
last_accepted_model_hash: <model hash>
```

The buffer is a thread-safe circular store backed by a sum tree. It samples
with replacement, returns normalized importance weights, and rejects stale
priority updates using slot generations.

### Game history and logs

`data/training_games.jsonl` contains one compact JSON object per native game.
Records include:

- `id`, `run_id`, `timestamp`, `batch_index`, `game_index`, and `game_number`;
- `result` as `1-0`, `0-1`, or `1/2-1/2`;
- `termination` such as `checkmate`, `draw`, `max_plies`, or `no_legal_move`;
- `moves` as standard UCI strings and `plies`;
- `experiences` and `target_counts`;
- `model_path` and the deployment `model_hash` when the sidecar is available.

The history writer rotates the file to `.1` after it exceeds 2 GiB. It does not
write PGN, SAN moves, Elo-after, Elo delta, loss, or training-step fields. The
frontend treats those fields as optional and displays defaults when they are
absent.

`logs/master_training.jsonl` contains training metrics. `logs/gate.jsonl`
contains gate results. `TrainingLogger` rotates its active log after 256 MiB.

## Model and training contract

### Network

The default `ModelConfig` is:

| Setting | Default |
| --- | ---: |
| Input channels | `121` |
| Trunk channels | `256` |
| Residual blocks | `12` |
| Policy size | `4672` |
| Policy head channels | `64` |
| SE reduction | `8` |
| Categorical value bins | `128` |

`ZeroNet` consists of a 3x3 convolutional stem, a stack of
`ConvResidualBlock` modules, and separate heads. The blocks use channel-first
LayerNorm, SiLU activations, two 3x3 convolutions, and a SwiGLU squeeze-and-
excitation gate. The heads are:

- policy logits over `4672` legal-move coordinates;
- a 128-bin categorical scalar value whose expected value lies in `[-1, 1]`;
- three-class WDL probabilities in win/draw/loss order;
- normalized white and black material auxiliaries;
- a normalized moves-left scalar;
- a dense opponent-policy auxiliary head retained for research experiments.

The opponent-policy head exists in the model and replay schema, but the
canonical `TrainConfig` sets its loss weight to zero and rejects a positive
weight because the target does not store the conditioning move needed for a
well-defined label.

### Input ABI

Both Python and Rust encode a board from the side-to-move perspective. White
uses the normal orientation. Black rotates the board 180 degrees and swaps
own/opponent piece groups. The tensor shape is `[N, 121, 8, 8]`.

The first `8 * 14 = 112` planes contain the current position followed by up to
seven previous positions, newest first. Each 14-plane slice contains:

| Planes | Meaning |
| --- | --- |
| `0..5` | Side-to-move's pawn, knight, bishop, rook, queen, and king |
| `6..11` | Opponent's pawn, knight, bishop, rook, queen, and king |
| `12` | Position has occurred at least twice |
| `13` | Position has occurred at least three times |

The final nine planes are:

| Offset | Meaning |
| ---: | --- |
| `112` | Side to move is White |
| `113` | Own kingside castling right |
| `114` | Own queenside castling right |
| `115` | Opponent kingside castling right |
| `116` | Opponent queenside castling right |
| `117` | En-passant file |
| `118` | `min(fullmove_number, 512) / 512` |
| `119` | `min(halfmove_clock, 100) / 100` |
| `120` | Side to move is in check |

Repetition counts are carried in the native replay payload because reparsing a
single FEN cannot reconstruct the sampled game's history. Training applies the
stored current and historical counts after FEN encoding.

### Policy ABI

The policy tensor has `73 * 64 = 4672` entries. An index is
`plane * 64 + oriented_from_square`.

| Planes | Encoding |
| --- | --- |
| `0..55` | Eight queen-like directions, seven distances per direction |
| `56..63` | Eight knight displacements |
| `64..72` | Knight, bishop, and rook underpromotions across three pawn directions |

Queen promotion uses the queen-like move encoding. Castling is exposed as
standard UCI (`e1g1`, `e1c1`, `e8g8`, `e8c8`) even though `cozy-chess` stores a
castling move internally as king-to-rook.

### TorchScript deployment ABI

`export_torchscript()` writes a module and a `<module>.sha256` sidecar. The
deployment wrapper accepts:

```text
x          [N, 121, 8, 8]       BF16 on CUDA, FP32 on CPU
legal_mask [N, 4672]            FP32, 1 for legal policy coordinates
```

It returns a TorchScript tuple:

```text
policy_logits [N, 4672]
scalar_value  [N, 1]
wdl           [N, 3]
```

The wrapper already fills illegal policy entries with a large negative value.
Rust applies the legal mask again before softmax as a defensive boundary and
returns legal probabilities, scalar values, and normalized WDL to MCTS.

### Training targets and loss

For terminal and adjudicated samples, the scalar target is:

```text
0.5 * (wdl_win - wdl_loss) + 0.5 * q_mcts
```

For `truncated` samples, the scalar target is `q_mcts` only. Truncated samples
retain a draw-shaped WDL vector for schema completeness, but their WDL loss is
masked to zero.

The default per-sample loss is:

```text
policy_loss
+ 0.25 * value_loss
+ 0.50 * wdl_loss
+ 0.05 * material_loss
+ 0.05 * moves_left_loss
+ 0.00 * opponent_policy_loss
```

The value target is converted to a two-hot distribution over the configured
value bins. The optimizer is AdamW with betas `(0.9, 0.95)`, weight decay
`1e-4`, an initial learning rate of `2e-3`, a final learning rate of `1e-5`,
linear warmup for 3,000 steps, cosine decay through 600,000 steps, and gradient
clipping at `1.0`. CUDA attempts fused AdamW and uses BF16 autocast when
supported. CPU paths use ordinary AdamW and no CUDA autocast.

Training updates replay priorities from absolute hybrid-value error plus policy
KL divergence. Horizontal board augmentation also reflects castling planes and
policy coordinates.

## Native self-play contract

The Rust engine is described in full in [RUST_ENGINE.md](RUST_ENGINE.md). The
production defaults used by `generate_self_play_batch_rust` are:

| Setting | Default |
| --- | ---: |
| Full-search simulations | `400` |
| Fast-search simulations | `35` |
| Probability of a full-search labelled ply | `0.20` |
| Fast-search policy target weight | `0.25` |
| Search batch size | `64` |
| Maximum game length | `512` plies |
| Rayon workers | Up to `24` |
| Node arena capacity per game | `250000` |

The Python bridge exposes the full-search simulation count, evaluator batch
size, and fast-search target weight. Fast-search budget, full-search probability,
maximum plies, and node capacity remain Rust defaults on the PyO3 production
entry point.

For each ply, Rust checks checkmate, stalemate/draw status, insufficient or
dead material, repetition, claimable repetition, and the halfmove threshold.
The first twelve plies use temperature `1.0`; later plies use temperature
`0.05`. Root Dirichlet noise is enabled for every training search. Every visited
position is recorded for replay; full-search policy targets use weight `1.0`,
while fast-search policy targets use the configurable reduced weight. Fast
searches therefore increase data throughput without being treated as equally
precise.

Terminal experiences carry a side-to-move WDL target and the final zero-sum
result. A game ending at `max_plies` is marked `target_kind = "truncated"` and
does not receive a terminal WDL training target. Native output also includes
history FENs, repetition counts, sparse policy visits, legal policy indices,
MCTS Q, material, moves-left, policy confidence, and an optional following
opponent policy.

## Interactive engine and API

### Python UCI

`zero-uci` is the public model-backed interactive engine. It uses the Python
board and Python MCTS, not the Rust production self-play MCTS.

With a checkpoint:

```bash
zero-uci \
  --checkpoint checkpoints/zero_x/accepted.pt \
  --device cuda \
  --simulations 800
```

Equivalent module invocation:

```bash
python -m zero_chess.uci \
  --checkpoint checkpoints/zero_x/accepted.pt \
  --device cuda \
  --simulations 800
```

Without `--checkpoint`, the engine intentionally uses `UniformEvaluator`; it
is a legal protocol fallback, not a trained model. UCI options are
`Simulations`, `CPuct`, `Checkpoint`, and `Device`. Supported command families
include `uci`, `isready`, `setoption`, `ucinewgame`, `position startpos`,
`position fen`, `go nodes`, `go depth`, `go movetime`, clock-based `go`,
`go infinite`, `stop`, `quit`, and `d`.

### Rust self-play facade

The thin Python facade is useful for generating one native batch outside the
master:

```bash
python -m zero_chess.self_play \
  --model checkpoints/zero_x/latest_model.ts \
  --games 200 \
  --simulations 160 \
  --batch-size 256 \
  --device cuda \
  --seed 1234 \
  --out data/self_play.json
```

This command requires the native extension and an exported TorchScript module.
It is not registered as a separate `pyproject.toml` console script.

### WebSocket server

`zero-ws` starts a FastAPI service that owns one recoverable Python UCI
subprocess:

```bash
zero-ws \
  --host 127.0.0.1 \
  --port 8765 \
  --checkpoint checkpoints/zero_x/accepted.pt \
  --device cuda \
  --simulations 200
```

The server falls back to CPU when CUDA is unavailable or lacks BF16 support.
The checkpoint is still required by default. It accepts origins listed in
`ZERO_ALLOWED_ORIGINS`; the default is:

```text
http://localhost:3000,http://127.0.0.1:3000
```

WebSocket endpoints are `/` and `/ws`. A request is JSON:

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "move_time": 1000
}
```

`move_time` is clamped to 100-120000 milliseconds. A successful response is:

```json
{
  "move": "e2e4",
  "evaluation": 0.12,
  "nodes": 200
}
```

Failures return `move: "0000"`, zeroed evaluation/nodes, and an `error`
string. Requests are serialized through one asyncio lock and one UCI process.
The HTTP endpoint `GET /history?limit=N` returns up to 500 recent records from
`data/training_games.jsonl` as `{ "games": [...] }`, ignoring malformed JSON
lines.

## Frontend

The frontend is a separate Next.js 14 application. Start the Python backend
first, then run:

```bash
cd frontend
npm ci
npm run dev
```

Available routes are:

| Route | Behavior |
| --- | --- |
| `/` | Navigation screen |
| `/play` | Play locally against the Python UCI/WebSocket engine |
| `/watch` | Watch both colors request moves from the same backend UCI process |
| `/history` | Poll and replay native JSONL game records |

The UI currently provides local move legality through `chess.js`, side
selection, 1/3/5/10-minute or unlimited clocks, promotion selection, board
flipping, move history, resignation, sounds, and engine-offline status. The
displayed ratings (`1500` for the user and `2450` for ZERO) are hard-coded UI
labels. They are not produced by the gate or a rating database.

The watch page is a sequential client of one engine process, not a separate
native two-engine tournament. The browser's shared `EngineSocket` permits one
pending request at a time. The backend likewise serializes access to its UCI
subprocess.

The history endpoint currently supplies UCI moves and provenance, not PGN or
training metrics. The history page accepts optional `pgn`, `moves_san`, Elo,
loss, and step fields for future producers, but the native writer does not
currently emit them. Multiplayer is displayed as disabled and is not
implemented. The frontend has no authentication, deployment orchestration, or
Next.js API proxy.

## Tests

The Python suite is configured in `pyproject.toml` with `tests/` as its test
path. The main coverage areas are:

| Test file | Coverage |
| --- | --- |
| `test_board.py` | FEN, legal moves, perft, castling, en passant, promotion, SAN, checks, mate, draws, repetition, hashing, undo |
| `test_encoding.py` | 121-plane shapes, perspective rotation, castling/en-passant planes, masks, batching |
| `test_encoding_mcts_uci.py` | Policy uniqueness, Python MCTS legality, UCI position and command flow |
| `test_mcts.py` | PUCT, FPU, batching, noise, virtual loss, reuse, resignation, reset, evaluator failures |
| `test_replay.py` | Prioritized sampling, weights, priorities, ring replacement, snapshots, RNG/cursor preservation |
| `test_rust_bridge.py` | Optional native board parity, sparse policy filtering, ingestion, material clamping, truncation, provenance |
| `test_model_optional.py` | Model shapes, legal masking, TorchScript wrapper parity, residual structure, optional CUDA inference |
| `test_training_optional.py` | Scheduler, training step, clipping, checkpoint round trip, hash validation, architecture recovery |
| `test_pipeline_hardening.py` | Replay edges, history, gating, run-state persistence, value bins, truncated targets |
| `test_audit_fixes.py` | Fast legal-move path, evaluator shutdown, checkpoint self-healing, targets, replay, MCTS, WebSocket parsing |
| `test_audit_fixes_v1287.py` | FPU compatibility, SE configuration recovery, virtual-loss constants |
| `test_hunter_rewards.py` | Zero-sum target and contempt regressions; no separate reward system |
| `test_zero_x.py` | Terminal targets, sign-alternating backup, WDL training, long-horizon schedule |
| `test_performance.py` | Prints a Python uniform-MCTS throughput record; asserts no performance threshold |

Rust unit tests are embedded in `src/encoding.rs`, `src/evaluator.rs`, and
`src/mcts.rs`. There is no tracked CI workflow, no frontend test suite, and no
pytest test that runs a full native TorchScript self-play batch; the preflight
script is the end-to-end native smoke test.

## Operational limitations

These are current system properties, not hidden assumptions:

1. **Native ABI compatibility is not automatic.** The required override for the
   locked `tch` and declared PyTorch versions disables a guard rather than
   validating the runtime ABI.
2. **CUDA requires BF16.** A GPU can be CUDA-capable and still be rejected by
   the model/export path if native BF16 support is unavailable.
3. **The Rust engine has fixed ceilings.** Evaluator batches are limited to 256
   leaves, native search/game depth is limited to 512 plies, and the default
   per-game node arena is 250,000 nodes. Arena exhaustion is an error.
4. **Fast-search policy targets are down-weighted.** Every ply enters replay,
   but fast-search policy targets use a lower confidence weight than full-search
   targets.
5. **Rust draw semantics are deliberately strict.** Claimable threefold and
   fifty-move positions terminate as draws in production self-play. Python's
   `Board.outcome(claim_draws=...)` exposes a configurable claim flag, but the
   native path has no equivalent decline-claims mode.
6. **The public UCI path is Python MCTS.** `zero-uci` and the WebSocket server
   do not invoke the production Rust self-play MCTS.
7. **The standalone Rust binary is uniform-only.** `zero-rust-engine` is useful
   for legal-move and protocol smoke tests, but it does not load TorchScript or
   represent trained model strength.
8. **Opponent-response supervision is inactive by default.** The head and
   schema are retained, but positive opponent-policy loss weight is rejected
   until the target includes the conditioning move.
9. **Truncated games are not relabelled as draws for WDL loss.** Their stored
   draw-shaped vector is schema-safe only; training uses MCTS Q for the scalar
   target and masks WDL loss.
10. **The WebSocket and browser clients are serialized.** One browser socket
    supports one pending request, and one server-side UCI process serves one
    request at a time.
11. **Frontend history is ahead of the backend schema.** The UI can display
    optional richer metadata, but the current native JSONL writer does not
    produce it.
12. **Performance tests are diagnostic only.** Unit-test throughput is not a
    deployment benchmark, and the repository makes no Elo or utilization claim.
13. **The restart wrappers retry every other non-zero exit.** A persisted
    deadline that has already expired makes `train_master.py` exit non-zero, so
    unattended wrappers should inspect and manage run state explicitly.
14. **Machine-local Cargo tuning is not the source contract.** A local ignored
    `.cargo/config.toml` may contain target-specific compiler or CUDA tuning; a
    fresh checkout does not receive those settings.

## License

ZERO-X is released under the [MIT License](LICENSE).
