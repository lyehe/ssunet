#!/bin/bash
# Two-phase training on corrected SPAD data
# Usage: bash train_corrected.sh <scene_name> <gpu_id>
# Example: bash train_corrected.sh blue-bathroom 0

SCENE=$1
GPU=${2:-0}
SCENE_DIR="D:/Yehe/PC/reconstruction/test/${SCENE}_corrected"
CKPT_DIR="models_spad"

echo "=== Phase 1: bf16-mixed, 200 epochs ==="
echo "Scene: $SCENE (corrected), GPU: $GPU"

uv run python run_spad.py \
    --scene-dir "$SCENE_DIR" \
    --device $GPU \
    --precision bf16-mixed \
    --max-epochs 200

echo "=== Phase 1 complete ==="

# Find the saved checkpoint from phase 1
PHASE1_CKPT="$CKPT_DIR/${SCENE}_corrected.ckpt"
if [ ! -f "$PHASE1_CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found at $PHASE1_CKPT"
    exit 1
fi

echo "=== Phase 2: fp32, 100 more epochs (300 total) ==="

uv run python run_spad.py \
    --scene-dir "$SCENE_DIR" \
    --device $GPU \
    --precision 32 \
    --max-epochs 300 \
    --resume "$PHASE1_CKPT"

echo "=== Phase 2 complete ==="
echo "Final checkpoint: $CKPT_DIR/${SCENE}_corrected.ckpt"
