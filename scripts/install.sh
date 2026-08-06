#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
if [[ "${ZERO_BUILD_NATIVE:-1}" == "1" ]]; then
    export LIBTORCH_USE_PYTORCH="${LIBTORCH_USE_PYTORCH:-1}"
    export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"
    cargo build --release --features libtorch,python-extension
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
python -m pytest --tb=short
