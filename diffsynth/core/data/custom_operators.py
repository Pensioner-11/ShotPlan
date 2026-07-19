import imageio
from PIL import Image
from diffsynth.core.data.operators import DataProcessingOperator


class LoadVideoRange(DataProcessingOperator):
    """Loads a frame range from a video file.

    Input:  dict {"path": str, "start_frame": int, "end_frame": int}
    Output: List[PIL.Image]
    """

    def __init__(self, frame_processor=lambda x: x):
        self.frame_processor = frame_processor

    def __call__(self, data: dict):
        path = data['path']
        start = data['start_frame']
        end = data['end_frame']

        target_count = end - start

        frames = []
        reader = None

        try:
            reader = imageio.get_reader(path)
            reader.set_image_index(start)

            for _ in range(target_count):
                try:
                    # get_next_data avoids a seek per frame.
                    frame = reader.get_next_data()
                    frame = Image.fromarray(frame)
                    frame = self.frame_processor(frame)
                    frames.append(frame)
                except (IndexError, RuntimeError, StopIteration):
                    break

        except Exception as e:
            print(f"[Warning] Failed to read video {path} at {start}: {e}")

        finally:
            if reader is not None:
                reader.close()

        current_len = len(frames)

        if 0 < current_len < target_count:
            # Short read near end of file: pad by repeating the last frame.
            print(f"[Warning] Padding video {path}: {current_len}/{target_count}")
            last_frame = frames[-1]
            for _ in range(target_count - current_len):
                frames.append(last_frame)

        elif current_len == 0:
            print(f"[Error] Skip corrupted video: {path}")
            return None

        return frames


class TailPadFrames(DataProcessingOperator):
    """Pads a frame list to `target_len` by repeating the last frame.

    WanVideo requires (4k + 1) frames on the temporal axis. Padding here,
    rather than reading extra source frames, guarantees the sample never
    crosses into the next shot of the source video.
    """

    def __init__(self, target_len: int):
        self.target_len = target_len

    def __call__(self, frames):
        if not frames:
            return None
        while len(frames) < self.target_len:
            frames.append(frames[-1])
        return frames
