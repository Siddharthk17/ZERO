"""Optional native extension build for source and wheel installations."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class NativeBuildExt(build_ext):
    """Build the PyO3 module when explicitly enabled by the installer."""

    def run(self) -> None:
        if os.environ.get("ZERO_BUILD_NATIVE", "0") != "1":
            return

        features = "libtorch,python-extension"
        environment = os.environ.copy()
        environment.setdefault("LIBTORCH_USE_PYTORCH", "1")
        if (
            environment.get("LIBTORCH_USE_PYTORCH") == "1"
            and environment.get("ZERO_ALLOW_UNSUPPORTED_LIBTORCH") != "1"
        ):
            raise RuntimeError(
                "PyTorch-backed native builds require ZERO_ALLOW_UNSUPPORTED_LIBTORCH=1 "
                "because tch 0.24.0 targets LibTorch 2.11.0"
            )
        if environment.get("ZERO_ALLOW_UNSUPPORTED_LIBTORCH") == "1":
            environment["LIBTORCH_BYPASS_VERSION_CHECK"] = "1"
        subprocess.run(
            ["cargo", "build", "--release", "--features", features, "--locked"],
            check=True,
            env=environment,
        )
        root = Path(__file__).resolve().parent
        artifact = next(
            (
                root / relative
                for relative in (
                    "target/release/libzero_rust_engine.so",
                    "target/release/zero_rust_engine.dll",
                    "target/release/libzero_rust_engine.dll",
                )
                if (root / relative).exists()
            ),
            None,
        )
        if artifact is None:
            raise RuntimeError("cargo completed without producing the PyO3 native extension")

        destination = Path(self.get_ext_fullpath("zero_chess.zero_rust_engine"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, destination)


setup(
    ext_modules=[Extension("zero_chess.zero_rust_engine", sources=[])],
    cmdclass={"build_ext": NativeBuildExt},
)
