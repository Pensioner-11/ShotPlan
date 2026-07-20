import json
import torch
import torch.nn as nn

from diffsynth.core.data.operators import ImageCropAndResize, DataProcessingOperator
from diffsynth.core.data.custom_operators import LoadVideoRange, TailPadFrames
from diffsynth.core.data.custom_units import WanVideoUnit_CutInjector
from diffsynth.core.data.custom_model_fn import model_fn_wan_video_with_cut


class BaseDataProfile:
    def __init__(self, args):
        self.args = args

    def load_and_transform(self, path):
        raise NotImplementedError

    def get_operator_map(self):
        return {}

    def get_data_keys(self):
        return []

    def get_extra_inputs(self):
        return []

    def configure_pipeline(self, pipe):
        pass


class PassThroughOp(DataProcessingOperator):
    def __call__(self, data):
        return data


class VidEventProfile(BaseDataProfile):
    """Data profile for ShotPlan multi-shot training samples.

    Expects a JSON list of records:
    {
        "file_path": "path/to/video.mp4",
        "start_frame": 102,
        "end_frame": 182,
        "cut_at": [26, 64],       # cut positions, frames relative to start_frame
        "type": "hardcut",
        "text": "Global caption ... Shot 1: ... Shot 2: ..."
    }
    """

    def load_and_transform(self, metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        flattened_data = []
        for row in raw_data:
            path = row['file_path']
            start = row.get('start_frame', 0)
            end = row.get('end_frame', -1)

            cut_at = row.get('cut_at', [])
            item_type = row.get('type', 'hardcut')

            item = {
                "video": {
                    "path": path,
                    "start_frame": start,
                    "end_frame": end
                },
                "cut_info": {
                    "cuts": cut_at,
                    "type": item_type
                },
                "prompt": row.get('text', '')
            }
            flattened_data.append(item)

        return flattened_data

    def get_operator_map(self):
        resize_op = ImageCropAndResize(
            height=self.args.height, width=self.args.width,
            max_pixels=self.args.max_pixels,
            height_division_factor=16, width_division_factor=16
        )
        # TailPadFrames pads the payload (e.g. 80 frames) to 4k+1 (e.g. 81)
        # by repeating the last frame, so the loader never reads past the
        # window into the next shot of the source video.
        return {
            "video": LoadVideoRange(frame_processor=resize_op) >> TailPadFrames(target_len=81),
            "cut_info": PassThroughOp()
        }

    def get_data_keys(self):
        return ["video", "cut_info"]

    def get_extra_inputs(self):
        return ["cut_info"]

    def configure_pipeline(self, pipe):
        print("[Profile] Configuring pipeline for planning-token injection...")

        # A single learnable planning token, hardcut_embedding, is used.
        dim = pipe.dit.dim

        def register_token(name):
            param_name = f"{name}_embedding"
            if not hasattr(pipe.dit, param_name):
                print(f"  -> Registering token: {param_name}")
                token_tensor = torch.randn(1, 1, dim) * 0.02
                cut_param = nn.Parameter(token_tensor)
                pipe.dit.register_parameter(param_name, cut_param)
                getattr(pipe.dit, param_name).requires_grad = True

        register_token("hardcut")

        # Insert the injector unit right after the noise initializer.
        pipe.units = [u for u in pipe.units if not isinstance(u, WanVideoUnit_CutInjector)]

        insert_index = 0
        for i, u in enumerate(pipe.units):
            if u.__class__.__name__ == "WanVideoUnit_NoiseInitializer":
                insert_index = i + 1
                break

        pipe.units.insert(insert_index, WanVideoUnit_CutInjector())

        pipe.model_fn = model_fn_wan_video_with_cut

        if hasattr(pipe.dit, "require_vae_embedding"):
            pipe.dit.require_vae_embedding = True

        print("[Profile] Pipeline configured.")
