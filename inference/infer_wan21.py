"""ShotPlan inference on Wan2.1-T2V-14B.

Generates multi-shot videos with hard cuts at user-specified frame indices.
A learnable planning token (trained together with the DiT) is injected into
the visual token sequence at a fractional temporal RoPE coordinate, one token
per requested cut.

Example:
    python inference/infer_wan21.py \
        --model_root /path/to/Wan2.1-T2V-14B \
        --ckpt /path/to/ShotPlan-Wan2.1-T2V-14B.safetensors \
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
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    x = latents

    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)

    x = dit.patchify(x, kwargs.get('control_camera_latents_input'))
    b, c, f, h, w = x.shape
    x = rearrange(x, 'b c f h w -> b (f h w) c')

    # --- Planning-token injection ---
    ids_h = torch.arange(h, device=x.device).repeat_interleave(w).repeat(f)
    ids_w = torch.arange(w, device=x.device).repeat(f * h)

    has_cut = cut_timestamps is not None and len(cut_timestamps) > 0 and getattr(dit, "hardcut_embedding", None) is not None

    if not has_cut:
        ids_f = torch.arange(f, device=x.device).repeat_interleave(h * w).float()

    if has_cut:
        if isinstance(cut_timestamps, torch.Tensor):
            cut_timestamps = cut_timestamps.tolist()
        sorted_cuts = sorted(cut_timestamps)

        cut_token = dit.hardcut_embedding.to(dtype=x.dtype)
        if cut_token.shape[0] != b:
            cut_token = cut_token.expand(b, -1, -1)

        x_segments = []
        f_segments = []
        h_segments = []
        w_segments = []

        for i in range(f):
            start_pos = i * h * w
            end_pos = (i + 1) * h * w
            x_segments.append(x[:, start_pos:end_pos])

            # Visual tokens keep integer temporal indices.
            current_f_ids = torch.full((h * w,), float(i), device=x.device, dtype=torch.float32)
            f_segments.append(current_f_ids)
            h_segments.append(ids_h[start_pos:end_pos])
            w_segments.append(ids_w[start_pos:end_pos])

            # Insert one token per cut timestamp in (i, i+1].
            cuts_in_gap = [ct for ct in sorted_cuts if i < ct <= (i + 1)]

            for cut_t in cuts_in_gap:
                x_segments.append(cut_token)
                f_segments.append(torch.tensor([cut_t], device=x.device, dtype=torch.float32))
                h_segments.append(torch.tensor([0], device=x.device))
                w_segments.append(torch.tensor([0], device=x.device))

        x = torch.cat(x_segments, dim=1)
        ids_f = torch.cat(f_segments)
        ids_h = torch.cat(h_segments)
        ids_w = torch.cat(w_segments)

    # --- 3D RoPE with fractional temporal coordinates ---
    table_h = dit.freqs[1].to(x.device)
    table_w = dit.freqs[2].to(x.device)
    emb_h = torch.nn.functional.embedding(ids_h, table_h)
    emb_w = torch.nn.functional.embedding(ids_w, table_w)

    if hasattr(dit, "num_heads"):
        num_heads = dit.num_heads
    elif hasattr(dit, "blocks") and len(dit.blocks) > 0 and hasattr(dit.blocks[0], "num_heads"):
        num_heads = dit.blocks[0].num_heads
    else:
        num_heads = dit.dim // 128

    head_dim = dit.dim // num_heads
    d_f = head_dim - 2 * (head_dim // 3)

    emb_f = compute_fractional_rope(ids_f.to(dtype=torch.float32), d_f)

    freqs = torch.cat([emb_f, emb_h, emb_w], dim=-1).unsqueeze(1)

    for block in dit.blocks:
        x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)

    # --- Remove the injected tokens (mirrors the injection loop) ---
    if has_cut:
        final_len = x.shape[1]
        keep_mask = torch.ones(final_len, dtype=torch.bool, device=x.device)
        current_ptr = 0

        for i in range(f):
            current_ptr += (h * w)
            cuts_in_gap = [ct for ct in sorted_cuts if i < ct <= (i + 1)]
            for _ in cuts_in_gap:
                if current_ptr < final_len:
                    keep_mask[current_ptr] = False
                current_ptr += 1

        x = x[:, keep_mask, :]

    x = dit.unpatchify(x, (f, h, w))
    return x


def load_pipeline(args, device):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=[
                f"{args.model_root}/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
                for i in range(1, 7)
            ]),
            ModelConfig(path=f"{args.model_root}/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=f"{args.model_root}/Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(path=f"{args.model_root}/google/umt5-xxl"),
    )

    # Register the planning token, then load the fine-tuned weights.
    dim = pipe.dit.dim
    pipe.dit.register_parameter(
        "hardcut_embedding",
        nn.Parameter(torch.zeros(1, 1, dim).to(device=device, dtype=torch.bfloat16))
    )

    print(f"Loading checkpoint from {args.ckpt}...")
    state_dict = load_safetensors(args.ckpt)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    msg = pipe.dit.load_state_dict(state_dict, strict=False)
    assert not msg.missing_keys, f"missing keys: {msg.missing_keys}"

    pipe.model_fn = model_fn_wan_t2v_with_cut
    return pipe


def run_task(pipe, task, args, output_dir):
    item_id, prompt, cut_at = task['id'], task['prompt'], str(task['cut_at'])
    output_path = os.path.join(output_dir, f"{item_id}_cut{cut_at.replace(',', '_')}.mp4")

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
    save_video(video_frames, output_path, fps=16, quality=5)
    print(f"Saved {output_path}")


def gpu_worker(gpu_id, task_queue, args):
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Initializing model on {device}...")

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
    parser = argparse.ArgumentParser(description="ShotPlan inference on Wan2.1-T2V-14B")
    parser.add_argument("--model_root", type=str, required=True, help="Wan2.1-T2V-14B model directory")
    parser.add_argument("--ckpt", type=str, required=True, help="ShotPlan fine-tuned .safetensors checkpoint")
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
        # Single-sample mode on one GPU.
        device = f"cuda:{args.gpus.split(',')[0]}"
        pipe = load_pipeline(args, device)
        task = {"id": "sample", "prompt": args.prompt, "cut_at": args.cut_at}
        if args.output:
            args.output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
            os.makedirs(args.output_dir, exist_ok=True)
            task["id"] = os.path.splitext(os.path.basename(args.output))[0].split("_cut")[0]
        run_task(pipe, task, args, args.output_dir)
        return

    if args.json_path is None:
        raise SystemExit("Provide either --prompt (single sample) or --json_path (batch).")

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    with open(args.json_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks.")

    task_queue = mp.Queue()
    for task in tasks:
        task_queue.put(task)

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
    for _ in gpu_list:
        task_queue.put(None)

    processes = []
    for gpu_id in gpu_list:
        p = mp.Process(target=gpu_worker, args=(gpu_id, task_queue, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All tasks completed.")


if __name__ == "__main__":
    main()
