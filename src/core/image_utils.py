from PIL import Image, ImageOps
import numpy as np
from pathlib import Path


STILL_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def load_image(image_path):
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        if img.format not in STILL_FORMATS:
            raise ValueError(
                f"Unsupported format: {img.format}. "
                "Only JPEG, PNG, GIF and WebP are supported."
            )
        if is_animated(img):
            frame, _duration = next(iter_animation_frames(img))
            return np.array(frame)
        oriented = orient_image(img)
        if oriented.mode != "RGB":
            oriented = oriented.convert("RGB")
        return np.array(oriented)


def save_image(image_array, output_path):
    img = Image.fromarray(image_array.astype(np.uint8))
    ext = Path(output_path).suffix.lower()
    kwargs = {}
    if ext in (".jpg", ".jpeg"):
        kwargs["quality"] = 95
    elif ext == ".webp":
        kwargs["quality"] = 90
    img.save(output_path, **kwargs)


def validate_image_format(file_path):
    ext = Path(file_path).suffix.lower()
    return ext in IMAGE_EXTENSIONS


def is_animated(image):
    return bool(getattr(image, "is_animated", False)) and getattr(image, "n_frames", 1) > 1


def orient_image(image):
    if is_animated(image):
        return image
    try:
        oriented = ImageOps.exif_transpose(image)
    except Exception:
        return image
    return image if oriented is None else oriented


def iter_animation_frames(image):
    total = getattr(image, "n_frames", 1)
    for frame_index in range(total):
        image.seek(frame_index)
        duration = int(image.info.get("duration", 100) or 100)
        frame = image.convert("RGB").copy()
        if frame.size != image.size:
            canvas = Image.new("RGB", image.size)
            canvas.paste(frame, (0, 0))
            frame = canvas
        yield frame, duration


def _gif_frames(frames):
    rgb_frames = [frame.convert("RGB") if frame.mode != "RGB" else frame for frame in frames]
    width, height = rgb_frames[0].size
    sample_count = min(len(rgb_frames), 6)
    if sample_count == 1:
        sample = rgb_frames[0]
    else:
        sample = Image.new("RGB", (width, height * sample_count))
        for index, frame in enumerate(rgb_frames[:sample_count]):
            if frame.size != (width, height):
                frame = frame.resize((width, height), Image.Resampling.BILINEAR)
            sample.paste(frame, (0, index * height))
    palette = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    return [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in rgb_frames]


def save_animation(frames, output_path, durations, loop=0, kind="GIF"):
    if not frames:
        raise ValueError("Animation contains no frames")
    if kind == "WEBP":
        first, *rest = frames
        first.save(
            output_path,
            format="WEBP",
            save_all=True,
            append_images=rest,
            duration=durations,
            loop=loop,
            quality=86,
        )
        return
    try:
        encoded = _gif_frames(frames)
    except Exception:
        encoded = [frame.convert("RGB") if frame.mode != "RGB" else frame for frame in frames]
    first, *rest = encoded
    first.save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=rest,
        duration=durations,
        loop=loop,
        disposal=1,
        optimize=False,
    )
