from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.diffusion.base_pipeline import PipelineUnit


class WanVideoUnit_CutInjector(PipelineUnit):
    """Turns per-sample cut annotations into a planning-token schedule.

    Consumes the `cut_info` field emitted by the dataset and produces the
    `cut_schedule` consumed by `model_fn_wan_video_with_cut`.
    """

    def __init__(self):
        super().__init__(
            input_params=("cut_info",),
            output_params=("cut_schedule",),
        )

    def process(self, pipe: WanVideoPipeline, cut_info):
        """
        Expected `cut_info` structure:
        {
            "cuts": [36, 43] or [15] or [[start, end], ...],
            "type": "hardcut" | "softcut" | "normal" | ...
        }

        Only 'hardcut' samples inject a token. Every element of 'cuts' becomes
        one hardcut_embedding at frame index f (mapped to the fractional
        latent coordinate t = 1 + f/4 for a 4x temporally compressed VAE).
        Other types (normal / softcut / camera motion) inject nothing so the
        model still sees them as regular video without any control token.
        """
        if cut_info is None:
            return {}

        raw_cuts = cut_info.get("cuts", [])
        cut_type = cut_info.get("type", "hardcut")

        if cut_type != "hardcut":
            return {"cut_schedule": []}

        def frame_to_t(f):
            return 1.0 + float(f) / 4.0

        schedule = []
        for f in raw_cuts:
            # Robust to both flat [f, ...] and nested [[f], ...] formats.
            val = f
            if isinstance(f, (list, tuple)):
                if len(f) == 0:
                    continue
                val = f[0]

            schedule.append({
                "t": frame_to_t(val),
                "token_name": "hardcut_embedding",
            })

        return {"cut_schedule": schedule}
