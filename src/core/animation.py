import numpy as np
from PIL import Image

from src.core.image_utils import is_animated, iter_animation_frames, orient_image
from src.core.face_swapper import FaceSequence


def animation_kind(image):
    return "WEBP" if image.format == "WEBP" else "GIF"


def process_animation(swapper, image, max_edge=1280, progress_cb=None, scale=0.75):
    source = image if is_animated(image) else orient_image(image)
    frames = []
    durations = []
    total = getattr(source, "n_frames", 1)
    sequence = FaceSequence(swapper)

    for index, (frame, duration) in enumerate(iter_animation_frames(source)):
        duration = int(duration or 0)
        rgb = np.asarray(frame.convert("RGB") if isinstance(frame, Image.Image) else frame)
        try:
            result = sequence.apply(rgb)
            if result.dtype != np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)
            if result.ndim != 3 or result.shape[2] != 3:
                raise ValueError("bad frame")
            frames.append(Image.fromarray(np.array(result, dtype=np.uint8, copy=True)))
        except Exception:
            frames.append(frame.copy() if isinstance(frame, Image.Image) else Image.fromarray(rgb))
        durations.append(duration if duration > 0 else 20)
        if progress_cb:
            progress_cb(index + 1, total)

    if not frames:
        raise ValueError("Animation contains no frames")
    return frames, durations, source.info.get("loop", 0), animation_kind(source)
