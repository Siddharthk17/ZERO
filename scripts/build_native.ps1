$ErrorActionPreference = "Stop"

if (-not $env:LIBTORCH_USE_PYTORCH) {
    $env:LIBTORCH_USE_PYTORCH = "1"
}
if (-not $env:LIBTORCH_BYPASS_VERSION_CHECK) {
    $env:LIBTORCH_BYPASS_VERSION_CHECK = "1"
}

cargo build --release --features libtorch,python-extension

$artifacts = @(
    "target/release/zero_rust_engine.dll",
    "target/release/libzero_rust_engine.dll"
)
$artifact = $artifacts | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $artifact) {
    throw "No Windows PyO3 DLL was produced by cargo build."
}

Copy-Item $artifact "zero_chess/zero_rust_engine.pyd" -Force
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import importlib; importlib.import_module('zero_chess.zero_rust_engine'); print('ZERO native extension ready')"
