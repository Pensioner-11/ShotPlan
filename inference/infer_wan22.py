"""ShotPlan inference on Wan2.2-T2V-A14B (MoE).

Wan2.2 routes denoising between two DiT experts: a high-noise model for early
steps and a low-noise model for late steps. Shot structure is decided early in
the denoising trajectory, so ShotPlan fine-tunes only the high-noise expert.
At inference the model_fn router below injects the planning token when the
active expert carries `hardcut_embedding` (the fine-tuned high-noise DiT) and
falls back to the stock DiffSynth model function for the untouched low-noise
expert.

Example:
    python inference/infer_wan22.py \
        --wan22_root /path/to/Wan2.2-T2V-A14B \
        --ckpt /path/to/ShotPlan-Wan2.2-T2V-A14B-HighNoise.safetensors \
        --prompt "Global caption ... Shot 1: ... Shot 2: ..." \
        --cut_at 40 \
        --output out.mp4
"""

import torch
import argparse
import os
import json
import torch.nn as nn
import torch.multiprocessing as mp
from einops import rearrange
from safetensors.torch import load_file as load_safetensors
from queue import Empty

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion.base_pipeline import PipelineUnit
from diffsynth.utils.data import save_video
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量"
)


def compute_fractional_rope(positions: torch.Tensor, dim: int, theta: float = 10000.0):
    """Fractional Temporal RoPE (FRoPE) for continuous positions."""
    freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device)[: (dim // 2)].double() / dim))
    freqs = torch.outer(positions.double(), freqs_base)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class WanVideoUnit_InferenceCut(PipelineUnit):
    """Converts user-specified cut frame indices into fractional latent timestamps."""

    def __init__(self, cut_at_str):
        super().__init__(input_params=(), output_params=("cut_timestamps",))
        self.cut_timestamps = self.parse_cut_at(cut_at_str)

    def parse_cut_at(self, cut_at):
        if not cut_at:
            return []
        frames = [int(x) for x in cut_at.split(",") if x.strip()] if isinstance(cut_at, str) else []
        # Frame f maps to latent coordinate t = 1 + f/4 (4x temporal VAE compression).
        return sorted(list(set([1.0 + f / 4.0 for f in frames if f >= 0])))

    def process(self, pipe):
        return {"cut_timestamps": self.cut_timestamps}


def model_fn_wan_t2v_with_cut(dit, latents, timestep, context, cut_timestamps=None, y=None, **kwargs):
    """Forward pass with planning-token injection (used by the high-noise expert)."""
    has_cut = hasattr(dit, "hardcut_embedding") and cut_timestamps is not None and len(cut_timestamps) > 0

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    x = latents

    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)

    x = dit.patchify(x, kwargs.get('control_camera_latents_input'))
    b, c, f, h, w = x.shape
    x = rearrange(x, 'b c f h w -> b (f h w) c')

    ids_h = torch.arange(h, device=x.device).repeat_interleave(w).repeat(f)
    ids_w = torch.arange(w, device=x.device).repeat(f * h)

    if not has_cut:
        ids_f = torch.arange(f, device=x.device).repeat_interleave(h * w).float()
    else:
        sorted_cuts = sorted(cut_timestamps)
        cut_token = dit.hardcut_embedding.to(dtype=x.dtype).expand(b, -1, -1)

        x_segments, f_segments, h_segments, w_segments = [], [], [], []
        for i in range(f):
            start_pos = i * h * w
            end_pos = (i + 1) * h * w
            x_segments.append(x[:, start_pos:end_pos])
            f_segments.append(torch.full((h * w,), float(i), device=x.device))
            h_segments.append(ids_h[start_pos:end_pos])
            w_segments.append(ids_w[start_pos:end_pos])

            cuts_in_gap = [ct for ct in sorted_cuts if i < ct <= (i + 1)]
            for cut_t in cuts_in_gap:
                x_segments.append(cut_token)
                f_segments.append(torch.tensor([float(cut_t)], device=x.device))
                h_segments.append(torch.tensor([0], device=x.device))
                w_segments.append(torch.tensor([0], device=x.device))

        x = torch.cat(x_segments, dim=1)
        ids_f = torch.cat(f_segments)
        ids_h = torch.cat(h_segments)
        ids_w = torch.cat(w_segments)

    table_h, table_w = dit.freqs[1].to(x.device), dit.freqs[2].to(x.device)
    emb_h, emb_w = torch.nn.functional.embedding(ids_h, table_h), torch.nn.functional.embedding(ids_w, table_w)

    num_heads = dit.num_heads if hasattr(dit, "num_heads") else (dit.dim // 128)
    head_dim = dit.dim // num_heads
    d_f = head_dim - 2 * (head_dim // 3)
    emb_f = compute_fractional_rope(ids_f, d_f)

    freqs = torch.cat([emb_f, emb_h, emb_w], dim=-1).unsqueeze(1)

    for block in dit.blocks:
        x = block(x, context, t_mod, freqs)
    x = dit.head(x, t)

    # Remove the injected tokens (mirrors the injection loop).
    if has_cut:
        final_len = x.shape[1]
        keep_mask = torch.ones(final_len, dtype=torch.bool, device=x.device)
        ptr = 0
        for i in range(f):
            ptr += (h * w)
            cuts_in_gap = [ct for ct in sorted_cuts if i < ct <= (i + 1)]
            for _ in cuts_in_gap:
                if ptr < final_len:
                    keep_mask[ptr] = False
                ptr += 1
        x = x[:, keep_mask, :]

    x = dit.unpatchify(x, (f, h, w))
    return x


def load_pipeline(args, device):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=[
                f"{args.wan22_root}/high_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
                for i in range(1, 7)
            ]),
            ModelConfig(path=[
                f"{args.wan22_root}/low_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
                for i in range(1, 7)
            ]),
            ModelConfig(path=f"{args.wan22_root}/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=args.vae_path or f"{args.wan22_root}/Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(path=f"{args.wan22_root}/google/umt5-xxl"),
    )

    # Register the planning token on the high-noise expert only, then load the
    # fine-tuned weights into it. The low-noise expert keeps original weights.
    high_dit = pipe.dit
    high_dit.register_parameter(
        "hardcut_embedding",
        nn.Parameter(torch.zeros(1, 1, high_dit.dim).to(device=device, dtype=torch.bfloat16))
    )

    print(f"Loading fine-tuned checkpoint into high-noise DiT: {args.ckpt}")
    state_dict = load_safetensors(args.ckpt)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    msg = high_dit.load_state_dict(state_dict, strict=False)
    assert not msg.missing_keys, f"missing keys: {msg.missing_keys}"

    # MoE routing: the fine-tuned high-noise expert (which carries
    # hardcut_embedding) goes through the injection model_fn; the stock
    # low-noise expert goes through the original DiffSynth model_fn.
    original_model_fn = pipe.model_fn

    def model_fn_router(dit, **kwargs):
        if hasattr(dit, "hardcut_embedding") and kwargs.get("cut_timestamps"):
            return model_fn_wan_t2v_with_cut(dit=dit, **kwargs)
        else:
            kwargs_for_native = {k: v for k, v in kwargs.items() if k != "cut_timestamps"}
            return original_model_fn(dit=dit, **kwargs_for_native)

    pipe.model_fn = model_fn_router
    return pipe


def run_task(pipe, task, args, output_dir):
    item_id, prompt, cut_at = task['id'], task['prompt'], str(task['cut_at'])
    output_path = os.path.join(output_dir, f"{item_id}_wan22_cut{cut_at.replace(',', '_')}.mp4")

    if os.path.exists(output_path):
        print(f"Skipping {item_id}, already exists.")
        return

    pipe.units = [u for u in pipe.units if not isinstance(u, WanVideoUnit_InferenceCut)]
    pipe.units.insert(0, WanVideoUnit_InferenceCut(cut_at_str=cut_at))

    video_frames = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        tiled=True,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps
    )
    save_video(video_frames, output_path, fps=15, quality=5)
    print(f"Saved {output_path}")


def gpu_worker(gpu_id, task_queue, args):
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading Wan2.2 MoE...")

    try:
        pipe = load_pipeline(args, device)
        print(f"[GPU {gpu_id}] Model loaded. Waiting for tasks...")
    except Exception as e:
        print(f"[GPU {gpu_id}] Init failed: {e}")
        import traceback
        traceback.print_exc()
        return

    while True:
        try:
            task = task_queue.get(timeout=5)
        except Empty:
            break
        if task is None:
            break

        try:
            run_task(pipe, task, args, args.output_dir)
        except Exception as e:
            print(f"[GPU {gpu_id}] Task {task.get('id')} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"[GPU {gpu_id}] Shutting down.")


def main():
    parser = argparse.ArgumentParser(description="ShotPlan inference on Wan2.2-T2V-A14B")
    parser.add_argument("--wan22_root", type=str, required=True, help="Wan2.2-T2V-A14B model directory")
    parser.add_argument("--vae_path", type=str, default=None, help="Path to Wan2.1_VAE.pth (defaults to <wan22_root>/Wan2.1_VAE.pth)")
    parser.add_argument("--ckpt", type=str, required=True, help="ShotPlan fine-tuned high-noise .safetensors checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results")
    # Single-sample mode
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for single-video generation")
    parser.add_argument("--cut_at", type=str, default="", help="Comma-separated cut frame indices, e.g. '26,64'")
    parser.add_argument("--output", type=str, default=None, help="Output mp4 path (single-sample mode)")
    # Batch mode
    parser.add_argument("--json_path", type=str, default=None, help="JSON task list: [{id, prompt, cut_at}, ...]")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids (batch mode)")
    # Sampling
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.prompt is not None:
        device = f"cuda:{args.gpus.split(',')[0]}"
        pipe = load_pipeline(args, device)
        task = {"id": "sample", "prompt": args.prompt, "cut_at": args.cut_at}
        if args.output:
            args.output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
            os.makedirs(args.output_dir, exist_ok=True)
            task["id"] = os.path.splitext(os.path.basename(args.output))[0].split("_wan22_cut")[0]
        run_task(pipe, task, args, args.output_dir)
        return

    if args.json_path is None:
        raise SystemExit("Provide either --prompt (single sample) or --json_path (batch).")

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    with open(args.json_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks.")

    task_queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
    for _ in gpu_list:
        task_queue.put(None)

    processes = [mp.Process(target=gpu_worker, args=(gid, task_queue, args)) for gid in gpu_list]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("All tasks completed.")


if __name__ == "__main__":
    main()
