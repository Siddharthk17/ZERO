# ZERO

ZERO is a self-born chess engine that learns only from its own self-play: it implements the full rules of chess, searches with batch MCTS guided by a Transformer-ResNet policy/value network, trains continuously after games, speaks UCI, and uses no human games, opening books, handcrafted evaluation, or tablebases.

## Hardware Requirements

Minimum target hardware is an RTX 2050-class NVIDIA GPU with 4GB VRAM. CPU mode is supported for smoke tests and bootstrapping, but useful training throughput requires CUDA.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python -c "import torch, zero_chess; print('ZERO ready')"
```

## Training

```bash
python train.py
```

## Resume Training

```bash
python train.py --resume checkpoints/latest.pt
```

## UCI Play

```bash
python -m zero_chess.uci
```

## Frontend

```bash
cd frontend && npm install
python -m zero_chess.websocket_server
cd frontend && npm run dev
```

## Lichess Deployment

```bash
cp configs/lichess-bot.yml.example lichess-bot.yml
lichess-bot --config lichess-bot.yml
```

## Expected Timeline

On an RTX 2050, generation 0 to 500 is expected to take approximately 2-3 days and should reach an estimated ELO range around 1400-1800, depending on simulation budget and thermal limits.

## Chess Core

The chess core owns board state, legal move generation, FEN, SAN, Zobrist hashing, repetition tracking, castling, en passant, promotion, checkmate, stalemate, fifty-move rule, and insufficient-material detection.

## Neural Network

The network encodes positions into 119 planes and evaluates them with a 20-block Transformer-ResNet tower, a masked 4672-logit policy head, a tanh value head, a WDL head, uncertainty output, and auxiliary prediction heads.

## MCTS Search

The search system uses PUCT, Dirichlet root noise for self-play, temperature-controlled move selection, virtual loss, transposition reuse, tree reuse, and batched neural evaluation.

## Continuous Learning

The learning system stores self-play experiences in prioritized hot and cold replay tiers, trains with policy, value, WDL, EWC, and auxiliary losses, updates an EMA teacher after each step, and checkpoints regularly.

## Self-Play And Arena

Self-play runs threaded games with opening diversity, symmetry augmentation, adjudication, and one online update per completed game batch, while arena evaluation tests the student against the teacher over balanced colors.

## Infrastructure

The infrastructure layer provides checkpoint management, replay persistence, PGN export, UCI compatibility, Lichess bot configuration, training logs, arena logs, and resumable `train.py` operation.

## License

MIT.
