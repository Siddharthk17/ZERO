#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python -c "import torch, zero_chess; print('ZERO ready')"
python -m pytest --tb=short
