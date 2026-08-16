#!/usr/bin/env bash
# Auto-restart wrapper for the canonical Rust self-play / Python training loop.
set -uo pipefail

CHECKPOINT="checkpoints/zero_x/accepted.pt"
RUN_STATE="checkpoints/zero_x/run_state.json"
REPLAY="checkpoints/zero_x/replay.pkl"
DEVICE="cuda"

echo "Starting ZERO Rust/Python master training loop..."
while true; do
    CMD=(python train_master.py --device "$DEVICE")
    if [ -f "$CHECKPOINT" ]; then
        echo "Resuming from existing checkpoint: $CHECKPOINT"
        CMD+=(--resume "$CHECKPOINT")
    elif [ -f "$RUN_STATE" ] || [ -f "$REPLAY" ]; then
        echo "Resuming from recoverable ZERO-X state"
    else
        echo "Starting training from scratch..."
        CMD+=(--fresh)
    fi
    
    # Run the canonical master loop and pass all CLI arguments through.
    set +e
    "${CMD[@]}" "$@"
    EXIT_CODE=$?
    set -e
    if [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 0 ]; then
        echo "Training interrupted or completed cleanly (exit code $EXIT_CODE)."
        break
    fi
    echo "Training process died with exit code $EXIT_CODE. Restarting in 5 seconds..."
    sleep 5
done
