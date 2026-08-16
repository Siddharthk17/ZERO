$ErrorActionPreference = "Stop"

if (-not $env:LIBTORCH_USE_PYTORCH) {
    $env:LIBTORCH_USE_PYTORCH = "1"
}
if ($env:LIBTORCH_USE_PYTORCH -eq "1" -and $env:ZERO_ALLOW_UNSUPPORTED_LIBTORCH -ne "1") {
    throw "PyTorch-backed native builds require ZERO_ALLOW_UNSUPPORTED_LIBTORCH=1 because tch 0.24.0 targets LibTorch 2.11.0."
}
if ($env:ZERO_ALLOW_UNSUPPORTED_LIBTORCH -eq "1") {
    $env:LIBTORCH_BYPASS_VERSION_CHECK = "1"
    Write-Warning "Building against an unchecked LibTorch/PyTorch ABI"
} else {
    Remove-Item Env:LIBTORCH_BYPASS_VERSION_CHECK -ErrorAction SilentlyContinue
}

cargo build --release --locked --features libtorch,python-extension

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
if ($env:ZERO_SKIP_PREFLIGHT -ne "1") {
    python scripts/preflight.py --device cpu --games 1 --simulations 1
}
