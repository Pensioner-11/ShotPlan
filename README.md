# ShotPlan: Cinematic Video Generation with Learnable Planning Token

Official implementation of **ShotPlan**, a framework for controllable multi-shot (cinematic) video generation built on Wan video diffusion models.

ShotPlan lets you specify **exact frame indices where hard cuts should happen** in a generated video. It introduces:

- **Learnable planning tokens** — a single learnable token, replicated per user-specified transition, is concatenated with the visual tokens and processed by the DiT as an in-context conditioning signal. No attention masks, no architectural surgery.
- **Fractional Temporal RoPE (FRoPE)** — video diffusion models operate in a temporally compressed latent space (4 physical frames per latent step), so cut timestamps rarely align with integer latent indices. Planning tokens are assigned *fractional* temporal RoPE coordinates (`t = 1 + frame/4`), enabling frame-level transition control while visual tokens keep the original pretrained RoPE.
- **High-noise-expert fine-tuning for Wan2.2 MoE** — shot structure is decided early in the denoising trajectory. For Wan2.2-T2V-A14B, only the high-noise expert is fine-tuned with planning tokens; the low-noise expert stays frozen and a model-function router dispatches between them at inference.

After denoising, the planning tokens are discarded — output shape and decoding are unchanged from the base model.


## Model Zoo

| Model | Base | HuggingFace |
|---|---|---|
| ShotPlan-Wan2.1-T2V-14B | [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | [Pensioner/ShotPlan-Wan2.1-T2V-14B](https://huggingface.co/Pensioner/ShotPlan-Wan2.1-T2V-14B) |
| ShotPlan-Wan2.2-T2V-A14B-HighNoise | [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) (high-noise expert) | [Pensioner/ShotPlan-Wan2.2-T2V-A14B-HighNoise](https://huggingface.co/Pensioner/ShotPlan-Wan2.2-T2V-A14B-HighNoise) |

Training data: [Pensioner/shotplan](https://huggingface.co/datasets/Pensioner/shotplan) (6.4K multi-shot samples curated from [VidEvent](https://arxiv.org/abs/2506.02448) with TransNet V2 + Gemini 2.5).

## Installation

```bash
git clone https://github.com/Pensioner-11/ShotPlan.git
cd ShotPlan
pip install -r requirements.txt
```

The repository vendors a trimmed copy of [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (`diffsynth/`, Apache-2.0) with ShotPlan's modifications in `diffsynth/core/data/custom_*.py` and `diffsynth/core/data/data_profiles.py`. Run all commands from the repository root (or add it to `PYTHONPATH`).

## Inference

### 1. Download base model weights

```bash
# Wan2.1-T2V-14B
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./models/Wan2.1-T2V-14B
# or Wan2.2-T2V-A14B
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./models/Wan2.2-T2V-A14B
```

### 2. Download ShotPlan checkpoints

```bash
huggingface-cli download Pensioner/ShotPlan-Wan2.1-T2V-14B --local-dir ./models/shotplan_wan21
huggingface-cli download Pensioner/ShotPlan-Wan2.2-T2V-A14B-HighNoise --local-dir ./models/shotplan_wan22
```

### 3. Generate

Prompts follow a hierarchical format: a global scene description followed by per-shot captions. `--cut_at` takes frame indices (81-frame video @ 16 fps).

**Wan2.1:**

```bash
python inference/infer_wan21.py \
    --model_root ./models/Wan2.1-T2V-14B \
    --ckpt ./models/shotplan_wan21/<checkpoint>.safetensors \
    --prompt "A rainy neon-lit street at night. Shot 1: Wide shot, a woman in a red coat walks toward the camera under an umbrella. Shot 2: Close-up of her face, rain drops on her cheek, neon reflections in her eyes." \
    --cut_at 40 \
    --output_dir ./results
```

**Wan2.2 (MoE):**

```bash
python inference/infer_wan22.py \
    --wan22_root ./models/Wan2.2-T2V-A14B \
    --ckpt ./models/shotplan_wan22/<checkpoint>.safetensors \
    --prompt "..." \
    --cut_at 26,64 \
    --output_dir ./results
```

Both scripts also support batch mode over multiple GPUs:

```bash
python inference/infer_wan21.py \
    --model_root ./models/Wan2.1-T2V-14B \
    --ckpt <ckpt> \
    --json_path tasks.json \      # [{"id": "...", "prompt": "...", "cut_at": "26,64"}, ...]
    --gpus 0,1,2,3
```

## Training

### Data

Download the dataset and point `METADATA` at the metadata JSON. Each record:

```json
{
  "file_path": "videos/V000001_16fps.mp4",
  "start_frame": 102,
  "end_frame": 182,
  "cut_at": [26, 64],
  "type": "hardcut",
  "text": "Global caption ... Shot 1: ... Shot 2: ..."
}
```

`cut_at` is in frames relative to `start_frame`. See the [dataset card](https://huggingface.co/datasets/Pensioner/shotplan) for details. To train on your own data, produce the same format (shot detection with TransNet V2 works well) and make `file_path` resolvable from `--dataset_base_path`.

### Launch

Wan2.1-T2V-14B, full-parameter, 8 GPUs (DeepSpeed ZeRO-2 + CPU offload; configs in `train/`):

```bash
WAN21_ROOT=./models/Wan2.1-T2V-14B \
METADATA=./data/train_meta_16fps.json \
bash train/train_wan21.sh
```

Wan2.2-T2V-A14B high-noise expert only:

```bash
WAN22_ROOT=./models/Wan2.2-T2V-A14B \
METADATA=./data/train_meta_16fps.json \
bash train/train_wan22_highnoise.sh
```

The planning token is registered as a DiT parameter named `hardcut_embedding` (shape `[1, 1, dim]`) and optimized jointly with the DiT weights. For Wan2.2, `--max_timestep_boundary 0.358` restricts training to the high-noise segment of the flow-matching schedule, matching the MoE routing boundary.

## How it works

```
video latents ── patchify ──► visual tokens (integer RoPE positions 0..f-1)
                                    │
user cut frames ──► fractional t = 1 + frame/4 ──► planning tokens (FRoPE, h=w=0)
                                    │
                    concatenate, sorted by time
                                    │
                              DiT blocks (unchanged)
                                    │
                    drop planning tokens ──► unpatchify ──► denoised latents
```

Key implementation files:

| File | What it does |
|---|---|
| `diffsynth/core/data/custom_model_fn.py` | Training-time model function: token injection, FRoPE, cleanup |
| `diffsynth/core/data/custom_units.py` | `WanVideoUnit_CutInjector`: cut annotations → token schedule |
| `diffsynth/core/data/data_profiles.py` | Dataset profile: registers `hardcut_embedding`, wires up the pipeline |
| `inference/infer_wan21.py` | Wan2.1 inference with cut control |
| `inference/infer_wan22.py` | Wan2.2 MoE inference: high-noise expert injects tokens, low-noise runs stock |

## Citation

```bibtex
@article{guo2026shotplan,
  title={ShotPlan: Cinematic Video Generation with Learnable Planning Token},
  author={Guo, Su and Liu, Guangce and Yang, Haosen and Wang, Jiepeng and Liu, Cong and Liu, Junqi and Huang, Haibin and Yao, Hongxun and Zhang, Chi and Li, Xuelong},
  year={2026}
}
```

If you use the training data, please also cite [VidEvent](https://arxiv.org/abs/2506.02448) (Liang et al.), from which our dataset is derived — we thank the authors for releasing it.

## Acknowledgements

- Built on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) by ModelScope.
- Base models: [Wan2.1 / Wan2.2](https://github.com/Wan-Video) by Alibaba.
- Training data derived from [VidEvent](http://www.videvent.top); shot detection by [TransNet V2](https://github.com/soCzech/TransNetV2).

## License

Apache-2.0, inherited from DiffSynth-Studio (see `LICENSE`). This repository contains modifications to the original DiffSynth-Studio code. Model weights inherit the Apache-2.0 license of the Wan model family. The dataset carries its own research-only terms — see the dataset card.
