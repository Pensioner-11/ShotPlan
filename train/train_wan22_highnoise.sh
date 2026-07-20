#!/bin/bash
# ShotPlan full-parameter training on the Wan2.2-T2V-A14B high-noise expert.
#
# Only the high-noise expert is fine-tuned: shot structure is decided early in
# the denoising trajectory, so the planning token is injected there while the
# low-noise expert stays frozen at its original weights. The timestep boundary
# below restricts training to the high-noise segment of the schedule.
#
# Usage:
#   WAN22_ROOT=/path/to/Wan2.2-T2V-A14B \
#   METADATA=/path/to/train_meta.json \
#   bash train/train_wan22_highnoise.sh

set -euo pipefail

WAN22_ROOT="${WAN22_ROOT:?Set WAN22_ROOT to the Wan2.2-T2V-A14B model directory}"
METADATA="${METADATA:?Set METADATA to the training metadata JSON}"
OUTPUT_PATH="${OUTPUT_PATH:-./checkpoints/shotplan_wan22_highnoise}"

export MASTER_PORT="${MASTER_PORT:-29501}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${SCRIPT_DIR}/accelerate_config_8gpu.yaml}"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  "${SCRIPT_DIR}/train.py" \
  --dataset_base_path . \
  --dataset_metadata_path "${METADATA}" \
  --data_format videvent \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_paths "[
    [
        \"${WAN22_ROOT}/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors\",
        \"${WAN22_ROOT}/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors\",
        \"${WAN22_ROOT}/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors\",
        \"${WAN22_ROOT}/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors\",
        \"${WAN22_ROOT}/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors\",
        \"${WAN22_ROOT}/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors\"
    ],
    \"${WAN22_ROOT}/models_t5_umt5-xxl-enc-bf16.pth\",
    \"${WAN22_ROOT}/Wan2.1_VAE.pth\"
    ]" \
  --tokenizer_path "${WAN22_ROOT}/google/umt5-xxl" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --num_epochs "${NUM_EPOCHS:-10}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-1}" \
  --dataset_num_workers "${DATASET_WORKERS:-8}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --trainable_models "dit" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0.0 \
  --find_unused_parameters \
  --num_frames 81 \
  --output_path "${OUTPUT_PATH}"
