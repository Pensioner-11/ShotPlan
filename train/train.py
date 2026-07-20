import torch, os, argparse, accelerate, warnings, json
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *

from diffsynth.core.data.custom_dataset import FlexibleDataset
from diffsynth.core.data.data_profiles import VidEventProfile

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROFILES = {
    "videvent": VidEventProfile,
}


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)

        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)

        if audio_processor_path is None:
            audio_processor_config = None
        else:
            audio_processor_config = ModelConfig(audio_processor_path)

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                # Pass any remaining fields straight through (e.g. cut_info).
                if extra_input in data:
                    inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1] if data.get("video") else 480,
            "width": data["video"][0].size[0] if data.get("video") else 832,
            "num_frames": len(data["video"]) if data.get("video") else 0,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="ShotPlan WanVideo Training Script")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to audio processor.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary.")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Init on CPU.")
    parser.add_argument("--data_format", type=str, default="videvent", help="Data profile key.")
    parser.add_argument("--ds_config", type=str, default=None, help="Path to DeepSpeed config JSON (default: ds_config.json next to this script).")
    parser.add_argument("--init_ckpt", type=str, default=None, help="Path to a trained checkpoint (step-N.safetensors) to warm-start the DiT from, e.g. to continue training after an interruption.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    ds_config_path = args.ds_config or os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_config.json")
    with open(ds_config_path) as f:
        ds_config = json.load(f)
    ds_plugin = accelerate.DeepSpeedPlugin(hf_ds_config=ds_config)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed_plugin=ds_plugin,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
        mixed_precision="bf16"
    )

    # --- Dataset ---
    current_profile = None
    extra_inputs_from_profile = []

    if args.data_format in PROFILES:
        if accelerator.is_main_process:
            print(f"[Main] Using data profile: {args.data_format}")

        current_profile = PROFILES[args.data_format](args)
        extra_inputs_from_profile = current_profile.get_extra_inputs()

        dataset = FlexibleDataset(
            data_profile=current_profile,
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
        )

    else:
        if accelerator.is_main_process:
            print(f"[Main] Using default UnifiedDataset logic.")

        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
        )

    # Merge extra inputs requested on the CLI with those required by the profile.
    user_extra_inputs = args.extra_inputs.split(",") if args.extra_inputs else []
    combined_extra_inputs = list(set(user_extra_inputs + extra_inputs_from_profile))
    final_extra_inputs_str = ",".join(combined_extra_inputs) if combined_extra_inputs else None

    if accelerator.is_main_process:
        print(f"[Main] Final extra_inputs: {final_extra_inputs_str}")

    # --- Model ---
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=final_extra_inputs_str,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )

    # Register the planning token and swap in the cut-aware model function.
    if current_profile is not None and hasattr(current_profile, "configure_pipeline"):
        current_profile.configure_pipeline(model.pipe)

    # Warm-start from a previously trained checkpoint (must happen after
    # configure_pipeline so hardcut_embedding exists as a parameter).
    if args.init_ckpt is not None:
        from safetensors.torch import load_file as load_safetensors
        if accelerator.is_main_process:
            print(f"[Main] Warm-starting DiT from checkpoint: {args.init_ckpt}")
        init_state_dict = load_safetensors(args.init_ckpt)
        load_msg = model.pipe.dit.load_state_dict(init_state_dict, strict=False)
        assert not load_msg.missing_keys, f"Missing keys when loading init_ckpt: {load_msg.missing_keys}"
        if accelerator.is_main_process and load_msg.unexpected_keys:
            print(f"[Main] init_ckpt unexpected keys (ignored): {load_msg.unexpected_keys}")
        del init_state_dict

    # --- Train ---
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }

    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
