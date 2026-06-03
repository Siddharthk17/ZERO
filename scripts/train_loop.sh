#!/usr/bin/env bash
# Auto-restart wrapper script for ZERO continuous training.
set -euo pipefail

CHECKPOINT="checkpoints/latest.pt"
DEVICE="cuda"

echo "Starting ZERO training loop..."
while true; do
    CMD="python train.py --device $DEVICE"
    if [ -f "$CHECKPOINT" ]; then
        echo "Resuming from existing checkpoint: $CHECKPOINT"
        CMD="$CMD --resume $CHECKPOINT"
    else
        echo "Starting training from scratch..."
    fi
    
    # Run train.py and pass all CLI arguments through
    if $CMD "$@"; then
        echo "Training finished cleanly."
        break
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 0 ]; then
            echo "Training interrupted or completed cleanly (exit code $EXIT_CODE)."
            break
        fi
        echo "Training process died with exit code $EXIT_CODE. Restarting in 5 seconds..."
        sleep 5
    fi
done
