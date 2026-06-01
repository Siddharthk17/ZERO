# ZERO: Self-Born Chess Reinforcement Learning Engine

> **A self-born chess engine that learns entirely from tabula rasa self-play.** ZERO implements the full rules of chess, searches using parallel batch MCTS guided by a hybrid Transformer-ResNet policy/value network, trains continuously, speaks UCI, and uses zero human games, opening books, handcrafted evaluations, or tablebases.

---

## System Architecture

![ZERO Chess System Architecture](zero-architecture-diagram.png)

ZERO's architecture is a highly optimized, dual-process pipeline designed to maximize both **GPU neural inference throughput** and **multiprocess CPU self-play generation** on consumer-grade hardware.

---

## Key Highlights & Core Principles

- **No Human Knowledge**: No handcrafted evaluation heuristics, opening books, or endgame tablebases. ZERO starts with zero knowledge of chess strategy beyond the basic movement rules.
- **Asymmetric Payoffs (Hunter Mode)**: Features aggressive reinforcement learning targets that penalize early resignation and stales draws to cultivate a highly active, attacking playstyle.
- **Hybrid Transformer-ResNet Policy/Value Network**: Combines deep convolutional residual towers (for local board spatial patterns) with Board Transformer Blocks (for global piece coordination).
- **Parallel Asynchronous Self-Play**: A persistent multi-process runtime where worker processes play chess games on CPU and stream evaluation queries to a single batching GPU worker.
- **Memory Guardrails**: Intelligently scales prioritized replay buffer capacity and monitors `/proc/meminfo` to auto-suspend training and release CUDA/RAM cache when resource thresholds are breached.

---

## Detailed Module Breakdown

### 1. Chess Core Rules Engine (`zero_chess/board.py`, `zero_chess/move.py`, `zero_chess/zobrist.py`)
- **Move Generation**: Custom rules engine with pseudo-legal and legal move generation for pawns, sliders, knights, castling, en passant, and promotions.
- **Incremental State Updates**: Highly optimized push/pop mechanics with an internal history stack.
- **Zobrist Hashing**: Collision-resistant 64-bit zobrist hash keys seeded deterministically, tracking en passant files, castling rights, active turns, and threefold repetitions.
- **Draw Detection**: Automatic verification of the 50-move rule, threefold repetition, and insufficient material (e.g. KVKB, KVKN, etc.).

### 2. Neural Network Architecture (`zero_chess/model.py`)
- **Input Feature Planes**: Encodes the board state into **119 planes** (consisting of the current and past 7 historical moves, castle rights, active turn, en passant squares, and move counts).
- **Residual Blocks**: Features Squeeze-and-Excitation (`SqueezeExcitation`) channels to scale feature relevance dynamically.
- **Global Attention**: Built-in global attention layers via `BoardTransformerBlock` to capture distant piece interactions.
- **Multi-Headed Outputs**:
  - **Policy Head**: Outputs masked logits representing the 4,672 policy coordinates.
  - **Value Head**: Evaluates board states with a scaled tanh activation representing $[-31.0, 1.0]$.
  - **WDL Head**: Predicts discrete Win, Draw, and Loss probabilities.
  - **Auxiliary Heads**: Predicts material balance, piece mobility, and king safety to speed up early-stage feature extraction.

### 3. Asymmetric Monte Carlo Tree Search (`zero_chess/mcts.py`)
- **PUCT Formula**: Guides search selection using node prior probabilities, visit counts, and virtual losses.
- **Dirichlet Noise**: Injects noise ($\alpha=0.3$, $\epsilon=0.25$) at the root node during self-play to guarantee opening diversity.
- **Subtree & Transposition Reuse**: Preserves visited child nodes across consecutive moves to execute zero-latency searches.
- **Iterative Node Clears**: Implements a robust iterative stack-based tree reset logic that completely eliminates recursion limits under deep search trees.

### 4. Prioritized Replay Buffer (`zero_chess/replay.py`)
- **Prioritized Replay (PER)**: Uses a binary SumTree for $O(\log N)$ priority updates and sampling.
- **Multi-Tier Replay Buffer**:
  - **Hot Tier**: Fast in-memory cache holding the latest high-priority experiences.
  - **Cold Tier**: Persistent SQLite3 database with Write-Ahead Logging (WAL) and index structures to manage millions of historical positions without high RAM footprints.
- **Hardware Autotuning**: Automatically decreases in-memory buffer capacities on systems with less than 2GB available RAM to prevent Out-Of-Memory (OOM) crashes.

### 5. Continuous Learning & Optimization (`zero_chess/training.py`, `zero_chess/ema.py`, `zero_chess/ewc.py`)
- **Continuous LRScheduler**: Cosine annealing decay driving rates from initial `1e-3` down to `3e-5`.
- **EWC (Elastic Weight Consolidation)**: Computes a quadratic weight penalty using the Fisher Information Matrix to prevent forgetting older opening patterns.
- **EMA Teacher**: Exponential Moving Average (`EMATeacher`) running updates on student weights ($\text{decay}=0.999$) to build a highly stable evaluator.

---

## Installation & Setup

### Hardware Requirements
- **VRAM**: RTX 2050 (or equivalent consumer laptop GPU) with at least **4GB VRAM**.
- **RAM**: Minimum **8GB System RAM** recommended (the replay engine automatically scales to fit tight systems).
- **CPU**: CPU mode is fully supported for testing, but active reinforcement training requires a CUDA-enabled GPU.

### Commands
```bash
# 1. Clone the repository and navigate to root
cd ZERO

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Install in developer mode
python -m pip install -e ".[dev]"

# 4. Verify PyTorch and CUDA availability
python -c "import torch, zero_chess; print('CUDA Ready:', torch.cuda.is_available())"
```

---

## Execution & Training Guide

### 1. Execute Unit Tests
Ensure the entire engine is fully functional by running the test suite:
```bash
python -m pytest
```

### 2. Start Self-Play Training Loop
Run the primary training coordinator:
```bash
python train.py
```
- **Bootstrap Phase**: The engine automatically starts with a CPU self-play bootstrap phase if no checkpoints are available, generating experiences using `UniformEvaluator`.
- **CUDA Multi-Process Phase**: Once bootstrap finishes, the loop spawns persistent worker processes that play games and enqueue positions asynchronously to the GPU-evaluator process.

### 3. Resume Training
To resume training from the latest auto-saved checkpoint:
```bash
python train.py --resume checkpoints/latest.pt
```

---

## Playing & Interface Integrations

### WebSocket Web GUI
ZERO comes with a premium frontend web application. To launch the Web GUI:
```bash
# 1. Launch the FastAPI WebSocket Server
python -m zero_chess.websocket_server --checkpoint checkpoints/latest.pt --device cuda

# 2. Build and launch the React Frontend
cd frontend
npm install
npm run dev
```

### UCI play
ZERO speaks the standard Universal Chess Interface protocol. You can connect it directly to any chess GUI (like Arena, ChessBase, or cutechess):
```bash
python -m zero_chess.uci --checkpoint checkpoints/latest.pt --device cuda
```

### Lichess Bot Deployment
To host your trained model on Lichess:
```bash
# 1. Copy the example configuration template
cp configs/lichess-bot.yml.example lichess-bot.yml

# 2. Edit lichess-bot.yml with your Lichess API Token and parameters
# 3. Run the Lichess Bot Bridge
lichess-bot --config lichess-bot.yml
```

---

## Algorithmic Payoff Matrices

ZERO enforces aggressive, non-zero-sum reinforcement learning payoffs designed to cultivate aggressive tactical behaviors:

| Outcome | My Payoff | Opponent Payoff | Strategic Rationale |
| :--- | :---: | :---: | :--- |
| **Checkmate Win** | `+1.0` | `-3.0` | Encourages hunting for absolute checkmate finishes. |
| **Checkmate Loss** | `-3.0` | `+1.0` | Severely penalizes getting checkmated. |
| **Resignation Win** | `0.0` | `-30.0` | Standard win value. |
| **Resignation Loss** | `-30.0` | `0.0` | Imposes catastrophic penalties on resigning early. |
| **Standard Draw** | `-1.0` | `-1.0` | Penalizes drawish peace offers. |
| **Stalemate Draw** | `-10.0` | `-10.0` | Discourages tactical stales. |
| **Max Plies Draw** | `-20.0` | `-20.0` | Severe penalty on long, repetitive endgames. |

### Symmetric Opponent Value Transform
To backpropagate these non-zero-sum payoffs correctly up MCTS paths, ZERO maps value perspectives using a symmetric piecewise linear interpolation passing through all major terminal coordinates:
```python
x0 <= val <= x1  =>  y0 + (val - x0) * (y1 - y0) / (x1 - x0)
```

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.
