import torch
from einops import rearrange

from typing import Optional, Dict, List
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.models.wan_video_vace import VaceWanModel
from diffsynth.models.wan_video_motion_controller import WanMotionControllerModel
from diffsynth.models.wan_video_animate_adapter import WanAnimateAdapter
from diffsynth.models.wan_video_mot import MotWanModel
from diffsynth.models.longcat_video_dit import LongCatVideoTransformer3DModel
from diffsynth.pipelines.wan_video import TeaCache, TemporalTiler_BCTHW, model_fn_longcat_video, model_fn_wans2v


def compute_fractional_rope(positions: torch.Tensor, dim: int, theta: float = 10000.0):
    """Fractional Temporal RoPE (FRoPE): RoPE rotations for continuous positions.

    Planning tokens are anchored between latent frames, so their temporal
    position is fractional. Standard RoPE lookup tables only cover integer
    indices; here the rotations are computed on the fly, in float64 to match
    the precision of the precomputed tables.

    Args:
        positions: (N,) temporal positions, fractional values allowed.
        dim: feature dimension allocated to the temporal axis of the 3D RoPE.
        theta: RoPE base frequency.
    Returns:
        (N, dim // 2) complex tensor of RoPE rotations.
    """
    freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device)[: (dim // 2)].double() / dim))
    freqs = torch.outer(positions.double(), freqs_base)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def model_fn_wan_video_with_cut(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    vap: MotWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    # Planning-token schedule, e.g. [{'t': 12.5, 'token_name': 'hardcut_embedding'}, ...]
    cut_schedule: List[Dict] = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    vap_hidden_state=None,
    vap_clip_feature=None,
    context_vap=None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    """WanVideo model function with planning-token (cut token) injection.

    Compared with the stock DiffSynth `model_fn_wan_video`, this version:
    1. Inserts one learnable planning token per entry of `cut_schedule` into
       the visual token sequence, between the latent frames that bracket the
       requested cut timestamp.
    2. Assigns each planning token a fractional temporal RoPE coordinate
       (FRoPE) so cuts are localized at frame level, with a fixed spatial
       coordinate (h, w) = (0, 0).
    3. Removes the planning tokens after the DiT blocks so the output shape
       matches the input latents.
    """
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
            cut_schedule=cut_schedule,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video_with_cut,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )

    if isinstance(dit, LongCatVideoTransformer3DModel):
        return model_fn_longcat_video(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            longcat_latents=longcat_latents,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep encoding
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    x = dit.patchify(x, control_camera_latents_input)

    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)

    b, c, f, h, w = x.shape
    x = rearrange(x, 'b c f h w -> b (f h w) c')

    # ------------------------------------------------------------------
    # Planning-token injection
    # ------------------------------------------------------------------
    ids_h = torch.arange(h, device=x.device).repeat_interleave(w).repeat(f)
    ids_w = torch.arange(w, device=x.device).repeat(f * h)

    has_cut = cut_schedule is not None and len(cut_schedule) > 0

    if not has_cut:
        ids_f = torch.arange(f, device=x.device).repeat_interleave(h * w).float()

    if has_cut:
        sorted_schedule = sorted(cut_schedule, key=lambda item: item['t'])

        x_segments = []
        f_segments = []
        h_segments = []
        w_segments = []

        for i in range(f):
            start_pos = i * h * w
            end_pos = (i + 1) * h * w
            x_segments.append(x[:, start_pos:end_pos])

            # Visual tokens keep their original integer temporal indices.
            current_f_ids = torch.full((h * w,), float(i), device=x.device, dtype=torch.float32)
            f_segments.append(current_f_ids)
            h_segments.append(ids_h[start_pos:end_pos])
            w_segments.append(ids_w[start_pos:end_pos])

            # Insert every planning token whose timestamp falls in (i, i+1].
            events_in_gap = [ev for ev in sorted_schedule if i < ev['t'] <= (i + 1)]

            for ev in events_in_gap:
                t_val = ev['t']
                token_name = ev['token_name']

                # The token parameter must be registered on the DiT
                # (see VidEventProfile.configure_pipeline).
                if not hasattr(dit, token_name):
                    continue

                token_param = getattr(dit, token_name).to(dtype=x.dtype)

                if token_param.shape[0] != b:
                    token_param = token_param.expand(b, -1, -1)

                x_segments.append(token_param)

                # Planning tokens get the fractional timestamp and a fixed
                # spatial coordinate (0, 0).
                f_segments.append(torch.tensor([t_val], device=x.device, dtype=torch.float32))
                h_segments.append(torch.tensor([0], device=x.device))
                w_segments.append(torch.tensor([0], device=x.device))

        x = torch.cat(x_segments, dim=1)
        ids_f = torch.cat(f_segments)
        ids_h = torch.cat(h_segments)
        ids_w = torch.cat(w_segments)

    # ------------------------------------------------------------------
    # 3D RoPE with fractional temporal coordinates
    # ------------------------------------------------------------------
    # Spatial axes use the precomputed lookup tables.
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

    # Temporal share of head_dim in WanVideo's 3D RoPE.
    d_f = head_dim - 2 * (head_dim // 3)
    emb_f = compute_fractional_rope(ids_f.to(dtype=torch.float32), d_f)

    freqs = torch.cat([emb_f, emb_h, emb_w], dim=-1).unsqueeze(1)

    # Reference image tokens (temporal position 0)
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)

        f_ref = 1
        ref_ids_f = torch.zeros(f_ref * h * w, device=x.device, dtype=torch.float32)
        emb_f_ref = compute_fractional_rope(ref_ids_f, d_f)

        ref_ids_h = torch.arange(h, device=x.device).repeat_interleave(w).repeat(f_ref)
        ref_ids_w = torch.arange(w, device=x.device).repeat(f_ref * h)
        emb_h_ref = torch.nn.functional.embedding(ref_ids_h, table_h)
        emb_w_ref = torch.nn.functional.embedding(ref_ids_w, table_w)

        freqs_ref = torch.cat([emb_f_ref, emb_h_ref, emb_w_ref], dim=-1).unsqueeze(1)
        freqs = torch.cat([freqs_ref, freqs], dim=0)

    # VAP
    if vap is not None:
        x_vap = vap_hidden_state
        x_vap = vap.patchify(x_vap)
        x_vap = rearrange(x_vap, 'b c f h w -> b (f h w) c').contiguous()
        clean_timestep = torch.ones(timestep.shape, device=timestep.device).to(timestep.dtype)
        t_vap = vap.time_embedding(sinusoidal_embedding_1d(vap.freq_dim, clean_timestep))
        t_mod_vap = vap.time_projection(t_vap).unflatten(1, (6, vap.dim))
        freqs_vap = vap.compute_freqs_mot(f, h, w).to(x.device)
        vap_clip_embedding = vap.img_emb(vap_clip_feature)
        context_vap = vap.text_embedding(context_vap)
        context_vap = torch.cat([vap_clip_embedding, context_vap], dim=1)

    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    if vace_context is not None:
        vace_hints = vace(
            x, vace_context, context, t_mod, freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload
        )

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]

    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            return lambda *inputs: module(*inputs)
        def create_custom_forward_vap(block, vap):
            return lambda *inputs: vap(block, *inputs)

        for block_id, block in enumerate(dit.blocks):
            if vap is not None and block_id in vap.mot_layers_mapping:
                args = (x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id)
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, x_vap = torch.utils.checkpoint.checkpoint(create_custom_forward_vap(block, vap), *args, use_reentrant=False)
                elif use_gradient_checkpointing:
                    x, x_vap = torch.utils.checkpoint.checkpoint(create_custom_forward_vap(block, vap), *args, use_reentrant=False)
                else:
                    x, x_vap = vap(block, *args)
            else:
                args = (x, context, t_mod, freqs)
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), *args, use_reentrant=False)
                elif use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), *args, use_reentrant=False)
                else:
                    x = block(x, context, t_mod, freqs)

            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale

            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)

        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x

    # ------------------------------------------------------------------
    # Remove the injected planning tokens.
    # The traversal below must mirror the injection loop exactly.
    # ------------------------------------------------------------------
    if has_cut:
        final_len = x.shape[1]
        keep_mask = torch.ones(final_len, dtype=torch.bool, device=x.device)
        curr_ptr = 0

        offset = reference_latents.shape[1] if reference_latents is not None else 0
        curr_ptr += offset

        sorted_schedule = sorted(cut_schedule, key=lambda item: item['t'])

        for i in range(f):
            curr_ptr += (h * w)

            events_in_gap = [ev for ev in sorted_schedule if i < ev['t'] <= (i + 1)]

            for ev in events_in_gap:
                if not hasattr(dit, ev['token_name']):
                    continue

                if curr_ptr < final_len:
                    keep_mask[curr_ptr] = False
                curr_ptr += 1

        x = x[:, keep_mask, :]

    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]

    x = dit.unpatchify(x, (f, h, w))

    return x
