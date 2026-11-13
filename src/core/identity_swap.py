import shutil
import threading
import urllib.request
from pathlib import Path

import numpy as np
import cv2

from src.core.paths import get_models_path

_MODEL_LOCK = threading.Lock()
_MODELS = {
    "w600k_r50.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.onnx",
        100_000_000,
    ),
    "inswapper_128.onnx": (
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx",
        400_000_000,
    ),
}


def _ensure_model(filename):
    path = Path(get_models_path(filename))
    url, minimum = _MODELS[filename]
    if path.is_file() and path.stat().st_size >= minimum:
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(filename + ".part")
    print(f"Downloading {filename}...", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "madurify"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {filename} into {path.parent}. "
            "Copy the model there and try again."
        ) from exc
    if not partial.is_file() or partial.stat().st_size < minimum:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Download of {filename} was incomplete.")
    partial.replace(path)
    return str(path)


def _identity_model_paths():
    with _MODEL_LOCK:
        return {name: _ensure_model(name) for name in _MODELS}

_ARCFACE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _similarity(src, dst):
    matrix, _ = cv2.estimateAffinePartial2D(
        np.asarray(src, dtype=np.float32),
        np.asarray(dst, dtype=np.float32),
        method=cv2.LMEDS,
    )
    return matrix


def _template(size):
    return _ARCFACE * (float(size) / 112.0)


def box_to_template(box, size=128):
    x, y, width, height = [float(value) for value in box]
    x -= width * 0.08
    y -= height * 0.18
    width *= 1.16
    height *= 1.55
    scale = (float(size) * 0.98) / max(height, 1.0)
    center_x = x + width * 0.5
    center_y = y + height * 0.46
    return np.array([
        [scale, 0.0, size * 0.50 - scale * center_x],
        [0.0, scale, size * 0.50 - scale * center_y],
    ], dtype=np.float32)


def image_to_template(keypoints, size=128):
    points = np.asarray(keypoints, dtype=np.float32).reshape(5, 2)
    destination = _template(size)
    eyes = points[:2]
    gap = float(np.linalg.norm(eyes[0] - eyes[1]))
    if gap >= 10.0:
        mouths = points[3:5]
        ordered = np.stack([
            eyes[np.argmin(eyes[:, 0])],
            eyes[np.argmax(eyes[:, 0])],
            points[2],
            mouths[np.argmin(mouths[:, 0])],
            mouths[np.argmax(mouths[:, 0])],
        ])
        return _similarity(ordered, destination)

    visible = eyes.mean(axis=0)
    nose = points[2]
    mouth = points[3:5].mean(axis=0)
    best_matrix = None
    best_score = -1e9
    for eye_index in (0, 1):
        source = np.stack([
            destination[eye_index],
            destination[2],
            (destination[3] + destination[4]) * 0.5,
        ])
        found = np.stack([visible, nose, mouth])
        template_to_image = _similarity(source, found)
        if template_to_image is None:
            continue
        image_to_box = cv2.invertAffineTransform(template_to_image)
        projected_eye = image_to_box[:, :2] @ visible + image_to_box[:, 2]
        projected_mouth = image_to_box[:, :2] @ mouth + image_to_box[:, 2]
        score = float(projected_mouth[1] - projected_eye[1])
        if projected_mouth[1] <= projected_eye[1]:
            score -= 1000.0
        if not (0.0 < projected_eye[0] < size and 0.0 < projected_mouth[1] < size):
            score -= 500.0
        if score > best_score:
            best_score = score
            best_matrix = image_to_box
    return best_matrix


class IdentitySwap:
    def __init__(self, reference_bgr, reference_keypoints):
        models = _identity_model_paths()
        emap_path = get_models_path("inswapper_emap.npy")
        swap_path = models["inswapper_128.onnx"]
        recog_path = models["w600k_r50.onnx"]
        self._emap = np.load(emap_path).astype(np.float32)
        self._recognizer = cv2.dnn.readNetFromONNX(recog_path)
        self._swapper = cv2.dnn.readNetFromONNX(swap_path)
        self._latent = self._embed(reference_bgr, reference_keypoints)

    @classmethod
    def from_faces(cls, faces, landmarks_list):
        best = 0
        best_span = -1.0
        for index, landmarks in enumerate(landmarks_list):
            points = np.asarray(landmarks, dtype=np.float64)
            if points.shape[0] < 68:
                continue
            span = float(np.linalg.norm(points[42:48].mean(axis=0) - points[36:42].mean(axis=0)))
            if span > best_span:
                best_span = span
                best = index
        landmarks = np.asarray(landmarks_list[best], dtype=np.float32)
        keypoints = np.stack([
            landmarks[36:42].mean(axis=0),
            landmarks[42:48].mean(axis=0),
            landmarks[30],
            landmarks[48],
            landmarks[54],
        ])
        image = np.asarray(faces[best])
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return cls(image, keypoints)

    def _embed(self, image_bgr, keypoints):
        matrix = image_to_template(keypoints, 112)
        if matrix is None:
            raise ValueError("Could not align the reference face")
        crop = cv2.warpAffine(
            image_bgr, matrix, (112, 112),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        blob = cv2.dnn.blobFromImage(
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
            1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=False,
        )
        self._recognizer.setInput(blob)
        embedding = self._recognizer.forward().reshape(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding) + 1e-6
        latent = (embedding @ self._emap).astype(np.float32)
        latent /= np.linalg.norm(latent) + 1e-6
        return latent

    def apply(self, image_rgb, keypoints, box=None):
        image = np.asarray(image_rgb)
        points = np.asarray(keypoints, dtype=np.float32).reshape(5, 2)
        eye_gap = float(np.linalg.norm(points[0] - points[1]))
        if box is not None and eye_gap < 10.0:
            matrix = box_to_template(box, 128)
        else:
            matrix = image_to_template(keypoints, 128)
        if matrix is None:
            return image
        aligned = cv2.warpAffine(
            image, matrix, (128, 128),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=False)
        self._swapper.setInput(blob, "target")
        self._swapper.setInput(self._latent.reshape(1, 512), "source")
        swapped = self._swapper.forward()[0].transpose(1, 2, 0)
        swapped = np.clip(swapped * 255.0, 0, 255).astype(np.uint8)

        height, width = image.shape[:2]
        inverse = cv2.invertAffineTransform(matrix)
        face = cv2.warpAffine(swapped, inverse, (width, height))
        mask = np.zeros((128, 128), dtype=np.float32)
        cv2.ellipse(mask, (64, 74), (54, 68), 0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 4.0)
        mask = cv2.warpAffine(mask, inverse, (width, height))
        alpha = mask[..., None]
        mixed = alpha * face.astype(np.float32) + (1.0 - alpha) * image.astype(np.float32)
        return np.clip(mixed, 0, 255).astype(np.uint8)
