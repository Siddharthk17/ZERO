"""Lichess deployment helpers and optimized bridge configuration generators."""

from __future__ import annotations


def default_uci_command(checkpoint: str = "checkpoints/latest.pt", device: str = "cuda", simulations: int = 800) -> str:
    """Return the standard command string used to launch the ZERO UCI engine on Lichess."""
    return f"python -m zero_chess.uci --checkpoint {checkpoint} --device {device} --simulations {simulations}"


def generate_lichess_config(
    token: str,
    bot_name: str = "ZERO_Bot",
    checkpoint: str = "checkpoints/latest.pt",
    device: str = "cuda",
    simulations: int = 800,
) -> str:
    """Generate the exact, optimized YAML configuration string required for the lichess-bot bridge.

    Tuned specifically for fast search speeds, high CPU-GPU batch synchronization, and absolute stability.
    """
    uci_cmd = default_uci_command(checkpoint, device, simulations)
    return f"""# ZERO Lichess Bot Deployment Configuration
token: "{token}"
url: "https://lichess.org"

engine:
  dir: "."
  name: "{uci_cmd}"
  protocol: "uci"
  silence_stderr: true
  concurrency: 1  # Standard single-process allocation for zero IPC overhead
  uci_options:
    Simulations: {simulations}
    CPuct: "1.5"
    Checkpoint: "{checkpoint}"
    Device: "{device}"

challenge:
  accept_bot: true
  only_rated: false
  time_controls:
    - bullet
    - blitz
    - rapid
    - classical
  modes:
    - casual
    - rated

matchmaking:
  allow_matchmaking: true
  challenge_interval: 30
  play_rate: 10
  opponent_min_rating: 600
  opponent_max_rating: 3000
"""