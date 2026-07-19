#!/bin/bash
# ShotPlan full-parameter training on Wan2.1-T2V-14B.
#
# Usage:
#   WAN21_ROOT=/path/to/Wan2.1-T2V-14B \
#   METADATA=/path/to/train_meta.json \
#   bash train/train_wan21.sh

set -euo pipefail

WAN21_ROOT="${WAN21_ROOT:?Set WAN21_ROOT to the Wan2.1-T2V-14B model directory}"
METADATA="${METADATA:?Set METADATA to the training metadata JSON}"
OUTPUT_PATH="${OUTPUT_PATH:-./checkpoints/shotplan_wan21_14b}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-shotplan_wan21_14b}"

export MASTER_PORT="${MASTER_PORT:-29531}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

accelerate launch \
  --config_file "${SCRIPT_DIR}/accelerate_config_8gpu.yaml" \
  "${SCRIPT_DIR}/train.py" \
  --dataset_base_path . \
  --dataset_metadata_path "${METADATA}" \
  --data_format videvent \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_paths "[
    [
        \"${WAN21_ROOT}/diffusion_pytorch_model-00001-of-00006.safetensors\",
        \"${WAN21_ROOT}/diffusion_pytorch_model-00002-of-00006.safetensors\",
        \"${WAN21_ROOT}/diffusion_pytorch_model-00003-of-00006.safetensors\",
        \"${WAN21_ROOT}/diffusion_pytorch_model-00004-of-00006.safetensors\",
        \"${WAN21_ROOT}/diffusion_pytorch_model-00005-of-00006.safetensors\",
        \"${WAN21_ROOT}/diffusion_pytorch_model-00006-of-00006.safetensors\"
    ],
    \"${WAN21_ROOT}/models_t5_umt5-xxl-enc-bf16.pth\",
    \"${WAN21_ROOT}/Wan2.1_VAE.pth\"
    ]" \
  --tokenizer_path "${WAN21_ROOT}/google/umt5-xxl" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --num_epochs "${NUM_EPOCHS:-10}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-1}" \
  --dataset_num_workers "${DATASET_WORKERS:-4}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --trainable_models "dit" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --max_timestep_boundary 1.0 \
  --min_timestep_boundary 0.0 \
  --find_unused_parameters \
  --output_path "${OUTPUT_PATH}"
