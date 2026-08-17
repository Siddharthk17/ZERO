#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e ".[dev]"
if [[ "${ZERO_BUILD_NATIVE:-1}" == "1" ]]; then
    export LIBTORCH_USE_PYTORCH="${LIBTORCH_USE_PYTORCH:-1}"
    if [[ "${LIBTORCH_USE_PYTORCH}" == "1" && "${ZERO_ALLOW_UNSUPPORTED_LIBTORCH:-0}" != "1" ]]; then
        echo "PyTorch-backed native builds require ZERO_ALLOW_UNSUPPORTED_LIBTORCH=1 because tch 0.24.0 targets LibTorch 2.11.0." >&2
        echo "Run the CPU/native preflight after enabling the explicit ABI override." >&2
        exit 2
    fi
    if [[ "${ZERO_ALLOW_UNSUPPORTED_LIBTORCH:-0}" == "1" ]]; then
        export LIBTORCH_BYPASS_VERSION_CHECK="1"
        echo "WARNING: building against an unchecked LibTorch/PyTorch ABI" >&2
    else
        unset LIBTORCH_BYPASS_VERSION_CHECK
    fi
    cargo build --release --locked --features libtorch,python-extension
    if [[ -f target/release/libzero_rust_engine.so ]]; then
        cp target/release/libzero_rust_engine.so zero_chess/zero_rust_engine.so
    elif [[ -f target/release/zero_rust_engine.dll ]]; then
        cp target/release/zero_rust_engine.dll zero_chess/zero_rust_engine.pyd
    elif [[ -f target/release/libzero_rust_engine.dll ]]; then
        cp target/release/libzero_rust_engine.dll zero_chess/zero_rust_engine.pyd
    else
        echo "Native build succeeded but no Python extension artifact was found." >&2
        exit 1
    fi
fi
python -c "import importlib, torch, zero_chess; importlib.import_module('zero_chess.zero_rust_engine'); print('ZERO native ready')"
if [[ "${ZERO_SKIP_PREFLIGHT:-0}" != "1" ]]; then
    python scripts/preflight.py --device cpu --games 1 --simulations 1
fi
python -m pytest --tb=short
