<div align="center">

# ShotPlan: Cinematic Video Generation with Learnable Planning Token

<p>
  <a href="https://pensioner-11.github.io/ShotPlan/"><img src="https://img.shields.io/badge/Project-Page-1f6feb?style=for-the-badge" alt="Project Page"></a>
  <a href="docs/assets/shotplan_paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b?style=for-the-badge" alt="Paper"></a>
  <a href="https://huggingface.co/Pensioner/ShotPlan-Wan2.2-T2V-A14B-HighNoise"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Hugging%20Face-ffcc4d?style=for-the-badge" alt="Model"></a>
  <a href="https://huggingface.co/datasets/Pensioner/shotplan"><img src="https://img.shields.io/badge/%F0%9F%97%82%EF%B8%8F%20Dataset-shotplan-4dc0b5?style=for-the-badge" alt="Dataset"></a>
</p>

**Su Guo**<sup>\*</sup> · **Guangce Liu**<sup>\*</sup> · Haosen Yang · Jiepeng Wang · Cong Liu · Junqi Liu · Haibin Huang · Hongxun Yao · Chi Zhang · Xuelong Li

</div>

---

ShotPlan equips a pre-trained text-to-video diffusion transformer with **learnable planning tokens** that place shot transitions at **user-specified frame indices** — frame-accurate hard cuts, smooth soft transitions, and temporally localized camera movement, all from a single text prompt with per-shot captions.

A single learnable embedding is replicated once per requested transition event and concatenated with the visual tokens. **Fractional Temporal RoPE (FRoPE)** gives each planning token a fractional latent-time coordinate derived from its target frame, so the token points *between* latent frames while the video tokens keep their original positions and the backbone's spatio-temporal prior is untouched. The tokens participate in standard self-attention, steer the denoising toward the requested shot structure, and are discarded before decoding.

## Model Zoo

| Model | Base | Hugging Face |
|---|---|---|
| **ShotPlan-Wan2.2-T2V-A14B-HighNoise** | Wan2.2-T2V-A14B (high-noise expert) | [Pensioner/ShotPlan-Wan2.2-T2V-A14B-HighNoise](https://huggingface.co/Pensioner/ShotPlan-Wan2.2-T2V-A14B-HighNoise) |
| ShotPlan-Wan2.1-T2V-14B | Wan2.1-T2V-14B | [Pensioner/ShotPlan-Wan2.1-T2V-14B](https://huggingface.co/Pensioner/ShotPlan-Wan2.1-T2V-14B) |

**Training data:** [Pensioner/shotplan](https://huggingface.co/datasets/Pensioner/shotplan) — 6.4K multi-shot samples curated from VidEvent with TransNet V2 + Gemini 2.5.

## How it works

<div align="center">
  <img src="docs/assets/figures/overview.png" alt="ShotPlan method overview" width="90%">
</div>

Latents are patchified as usual; for a request like hard cuts at frames 21, 46, 64, three copies of the learnable planning embedding are appended with fractional temporal positions 6.00, 12.25, 16.75. They attend jointly with the visual tokens through every DiT block and are removed before unpatchify.

## Inference

We recommend using the **Wan2.2** model for better results.

1. **Download a base model** — [Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) (recommended) or [Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B).
2. **Download the matching ShotPlan checkpoint** — [ShotPlan-Wan2.2-T2V-A14B-HighNoise](https://huggingface.co/Pensioner/ShotPlan-Wan2.2-T2V-A14B-HighNoise) (recommended) or [ShotPlan-Wan2.1-T2V-14B](https://huggingface.co/Pensioner/ShotPlan-Wan2.1-T2V-14B).
3. **Generate.** Prompts follow a hierarchical format: a global scene description followed by per-shot captions. `--cut_at` takes comma-separated frame indices (81-frame video @ 16 fps).

```bash
git clone https://github.com/Pensioner-11/ShotPlan.git && cd ShotPlan
pip install -r requirements.txt

python inference/infer_wan22.py \
    --wan22_root ./models/Wan2.2-T2V-A14B \
    --ckpt ./models/shotplan_wan22/ShotPlan-Wan2.2-T2V-A14B-HighNoise.safetensors \
    --prompt "Global scene description. Shot 1: ... Shot 2: ... Shot 3: ..." \
    --cut_at 21,46,64 \
    --output_dir ./results
```

## Data

The training set is released on Hugging Face: **[Download the ShotPlan dataset](https://huggingface.co/datasets/Pensioner/shotplan)** — 6.4K multi-shot samples (480×832, 81 frames @ 16 fps) curated from VidEvent, with shot boundaries verified by TransNet V2 and per-shot captions written by Gemini 2.5.

## Citation

```bibtex
@article{guo2026shotplan,
  title={ShotPlan: Cinematic Video Generation with Learnable Planning Token},
  author={Guo, Su and Liu, Guangce and Yang, Haosen and Wang, Jiepeng and Liu, Cong and Liu, Junqi and Huang, Haibin and Yao, Hongxun and Zhang, Chi and Li, Xuelong},
  year={2026}
}
```

## License

Apache-2.0, inherited from the Wan base models.
