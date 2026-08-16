# ZERO-X: Hardware-Aware Tabula-Rasa Chess RL

ZERO-X is a pure self-play chess research engine. It uses no human games,
opening books, handcrafted evaluation terms, pretrained weights, or tablebases.
The only built-in knowledge is the legal rules of chess and the policy-coordinate
mapping required to represent legal moves.

## Learning objective

Every terminal training target is strictly zero-sum:

| Result | White | Black | WDL target |
| --- | ---: | ---: | --- |
| `1-0` | +1 | -1 | win / loss |
| `0-1` | -1 | +1 | loss / win |
| `1/2-1/2` | 0 | 0 | draw / draw |

MCTS backs up values by exact sign alternation at every ply. Search-time
contempt is optional and bounded; it is never written into replay data or used
as a policy or value target. There is no early model-value adjudication or
capture/aggression/mobility/king-safety reward shaping. Normalized material is
an auxiliary supervised label only; it is not a game reward or evaluator term.

## Architecture

- **Network:** 12-block, 256-channel Squeeze-and-Excitation ResNet with a
  73-plane policy head, WDL distribution, scalar value, material,
  moves-left, and sparse opponent-response auxiliary heads.
- **Inputs:** 121 planes: eight position slices (piece and repetition planes)
  plus side-to-move, castling, en-passant, fullmove, halfmove, and check state.
- **Search:** batched PUCT with virtual loss, first-play urgency, root
  Dirichlet noise for self-play, and tree reuse.
- **Training:** fused AdamW, BF16 autocast, policy/WDL loss plus a 50:50
  terminal-outcome/MCTS-Q value target, normalized material and moves-left
  targets, gradient clipping, RAM-only prioritized replay, warmup, and cosine
  decay.  PCR uses 20% full-simulation labelled plies and 80% fast-simulation
  progression plies (default 35 simulations).
- **Runtime:** Rust owns production self-play, move generation, MCTS, and
  fixed-slot CUDA evaluation. Python owns replay ingestion, model updates,
  checkpoints, and TorchScript export.

The CUDA training path enables TF32 where applicable and BF16 autocast. Actual
throughput is hardware and workload dependent; record it for each experiment
rather than inferring it from a configuration target.

## Repository layout

```text
src/                 Rust production core: cozy-chess, flat MCTS arena, CUDA actor, PyO3
zero_chess/model.py  SE-ResNet, WDL inference ABI, TorchScript export
zero_chess/replay.py RAM-only prioritized replay
zero_chess/rust_bridge.py  Sparse native experience ingestion and validation
zero_chess/self_play.py   Thin Rust-only CLI facade; no Python workers
train_master.py     Canonical Rust self-play → WDL training → candidate gate loop
```

The Python board/MCTS/UCI/arena modules are retained only for deterministic
tests and isolated evaluation. The master loop never uses them for production
self-play.

## Running a fresh experiment

The Python package installs without compiling native code by default. For a
production native install on the workstation, use `scripts/install.sh`, or
explicitly set `ZERO_BUILD_NATIVE=1` when building a wheel. If the installed
PyTorch version is newer than `tch`, the ABI check must be consciously enabled
with `ZERO_ALLOW_UNSUPPORTED_LIBTORCH=1` only after native smoke validation.
The locked `tch 0.24.0` binding targets LibTorch 2.11.0, while the target
workstation uses PyTorch 2.12. The override is therefore required for this
repository version and must be followed by `scripts/preflight.py --full-model`;
do not treat a successful compile alone as ABI validation.

For the declared workstation stack, the native install command is:

```bash
ZERO_ALLOW_UNSUPPORTED_LIBTORCH=1 bash scripts/install.sh
```

ZERO-X stores its outputs separately from retired checkpoints and replay data:

- checkpoints: `checkpoints/zero_x/`
- replay: memory only during active training (optional explicit snapshots)

```bash
python -m pytest
python scripts/perft.py --depth 5
python scripts/preflight.py --device cuda
# On the target workstation, also run the full production model smoke test.
python scripts/preflight.py --device cuda --full-model
bash scripts/train_loop.sh
# Windows PowerShell equivalent:
.\scripts\train_loop.ps1 -Device cuda -Days 31
```

The training deadline is persisted in `checkpoints/zero_x/run_state.json`, so
the restart wrappers resume the original 31-day window instead of starting a
new window after a crash.

`perft.py --depth 5` must report `4865609`. The restart wrapper prefers
`checkpoints/zero_x/accepted.pt` and falls back only to validated local
checkpoints, preventing accidental use of pre-ZERO-X Transformer/reward-shaped
checkpoints.

Useful launch parameters:

```bash
python train_master.py --device cuda \
  --games-per-batch 200 --simulations 400 --eval-batch-size 256
# Add --fresh for a strict run that refuses existing checkpoints/replay.
```

Candidate checkpoints are gated against the last accepted model before they
are exported to native self-play. Use `--disable-gating` only for smoke tests.

## Evaluation

The master loop compares each candidate against the accepted checkpoint and
exports it to self-play only after the gate's confidence interval is above
50%. For clean throughput measurements, run larger matches separately. Elo and
throughput targets must be measured on the actual deployment hardware;
configuration alone is not evidence of either.

## Research framing

Suggested thesis title: *ZERO-X: Hardware-Aware Single-Workstation Tabula-Rasa
Reinforcement Learning for Autonomous Chess Discovery.*

Report measured positions/sec, evaluations/sec, update time, GPU utilization,
SPRT outcomes, and tactical/positional test performance; do not present
configuration targets as experimentally established results.
