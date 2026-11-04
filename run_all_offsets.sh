#!/bin/bash
# Run 16-offset inference for all scenes with trained checkpoints.
# Usage: bash run_all_offsets.sh <gpu_id>
# GPU 0 gets first half, GPU 1 gets second half.

GPU=${1:-0}

declare -A SCENES
# scene_name -> "scene_dir checkpoint"
SCENES[attic]="D:/Yehe/PC/reconstruction/train/attic models_spad/attic.ckpt"
SCENES[basement]="D:/Yehe/PC/reconstruction/test/basement models_spad/basement.ckpt"
SCENES[bedroom-interior]="D:/Yehe/PC/reconstruction/test/bedroom-interior models_spad/bedroom-interior.ckpt"
SCENES[blue-bathroom_corrected]="D:/Yehe/PC/reconstruction/test/blue-bathroom_corrected models_spad/blue-bathroom_corrected.ckpt"
SCENES[coffee-office-bar]="D:/Yehe/PC/reconstruction/test/coffee-office-bar models_spad/coffee-office-bar.ckpt"
SCENES[train-passby_corrected]="D:/Yehe/PC/reconstruction/test/train-passby_corrected models_spad/train-passby_corrected.ckpt"

if [ "$GPU" = "0" ]; then
    ORDER=(attic basement bedroom-interior)
else
    ORDER=(blue-bathroom_corrected coffee-office-bar train-passby_corrected)
fi

for SCENE in "${ORDER[@]}"; do
    IFS=' ' read -r SCENE_DIR CKPT <<< "${SCENES[$SCENE]}"
    echo "=== $SCENE on GPU $GPU ==="
    uv run python infer_spad_offsets.py \
        --scene-dir "$SCENE_DIR" \
        --ckpt "$CKPT" \
        --device "$GPU" \
        --chunk 0 \
        --output-dir "output/$SCENE/offsets"
done
