# zero-rust-engine

The Rust core uses `cozy-chess` for legal move generation and a flat, fixed-capacity MCTS arena. Each Rayon self-play worker owns one arena; all workers submit fixed CPU slabs to one CUDA batch actor. No tree node is heap allocated after an arena is created.

## Build

The CPU development build and tests need no LibTorch:

```bash
cargo test --lib
```

For the Python extension and the CUDA evaluator, build against a LibTorch distribution matching the `tch` version in `Cargo.toml`:

```bash
export LIBTORCH=/opt/libtorch
cargo build --release --locked --features libtorch,python-extension
ln -sf target/release/libzero_rust_engine.so zero_rust_engine.so
```

When intentionally building against a Python PyTorch install, `tch` supports `LIBTORCH_USE_PYTORCH=1`. Its ABI version check is enforced by default. Set `ZERO_ALLOW_UNSUPPORTED_LIBTORCH=1` only after validating the exact PyTorch/LibTorch ABI for the target deployment:

```bash
export LIBTORCH_USE_PYTORCH=1
export ZERO_ALLOW_UNSUPPORTED_LIBTORCH=1
cargo build --release --locked --features libtorch,python-extension
```

The override is mandatory for the repository's locked `tch 0.24.0` versus the
PyTorch 2.12 workstation stack. Run the native preflight after every clean
installation. TorchScript exports also write a `.sha256` sidecar used for
deployment provenance and evaluator cache invalidation.

On Windows, run `powershell -ExecutionPolicy Bypass -File
scripts/build_native.ps1`. The script installs the generated DLL as
`zero_chess/zero_rust_engine.pyd`, which is the Python extension suffix needed
by the Windows interpreter.

`target-cpu=native` and Blackwell `TORCH_CUDA_ARCH_LIST=12.0` are set in `.cargo/config.toml`.

## Export and call from Python

Python checkpoints are state dictionaries, not executable LibTorch modules. Export a deployment module after each accepted training checkpoint:

```python
from zero_chess.model import export_torchscript, load_model

model = load_model("checkpoints/zero_x/accepted.pt", "cuda")
export_torchscript("checkpoints/zero_x/accepted.ts", model, "cuda")

from zero_rust_engine import generate_self_play_batch_rust

batch = generate_self_play_batch_rust(
    "checkpoints/zero_x/accepted.ts", num_games=24, simulations=400,
    batch_size=256, device="cuda", seed=1234,
)
```

The exported module returns raw policy logits; the Rust actor applies its compact legal mask and softmax on CUDA. The returned dictionary has `games`; each game has a result, termination reason, UCI moves, and replay-compatible experience dictionaries (`fen`, sparse `policy_indices`/`policy_values`, `target_kind`, scalar value, WDL, and history FENs). The sparse policy avoids transferring 4,672 Python floats per position; `zero_chess.rust_bridge` converts it to replay's legal UCI policy map.

`FastRustBoard` is an interactive native board wrapper. `analyze_uniform()` is a diagnostic smoke search; it intentionally does not claim model strength without a TorchScript network.

Replay experiences include `target_kind`. `terminal` is a chess-terminal target;
`truncated` means the configured game length was reached without a terminal
result and must not be trained as a WDL draw.
