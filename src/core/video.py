import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import dlib
import numpy as np

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def is_video_file(path):
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


class LandmarkSmoother:

    def __init__(self, alpha=0.65, max_misses=8):
        self.alpha = alpha
        self.max_misses = max_misses
        self.tracks = {}
        self._next_id = 0

    @staticmethod
    def _center(rect):
        return np.array([(rect.left() + rect.right()) / 2.0,
                         (rect.top() + rect.bottom()) / 2.0])

    def update(self, detections):
        used = set()
        output = []

        for rect, landmarks in detections:
            center = self._center(rect)
            radius = max(rect.width(), rect.height()) * 0.75

            best_id, best_dist = None, float('inf')
            for tid, track in self.tracks.items():
                if tid in used:
                    continue
                dist = np.linalg.norm(center - self._center(track['rect']))
                if dist < radius and dist < best_dist:
                    best_id, best_dist = tid, dist

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self.tracks[best_id] = {'rect': rect, 'landmarks': landmarks.astype(np.float64),
                                        'misses': 0}
            else:
                track = self.tracks[best_id]
                a = self.alpha
                sm_lm = a * landmarks + (1.0 - a) * track['landmarks']
                r, tr = rect, track['rect']
                sm_rect = dlib.rectangle(
                    int(a * r.left() + (1 - a) * tr.left()),
                    int(a * r.top() + (1 - a) * tr.top()),
                    int(a * r.right() + (1 - a) * tr.right()),
                    int(a * r.bottom() + (1 - a) * tr.bottom()),
                )
                track['rect'] = sm_rect
                track['landmarks'] = sm_lm
                track['misses'] = 0

            used.add(best_id)
            track = self.tracks[best_id]
            output.append((track['rect'], track['landmarks']))

        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]['misses'] += 1
                if self.tracks[tid]['misses'] > self.max_misses:
                    del self.tracks[tid]

        return output

    def current(self):
        return [(t['rect'], t['landmarks']) for t in self.tracks.values()]


class FrameSwapEngine:

    def __init__(self, swapper, scale=0.6, detect_every=3, smooth_alpha=0.65):
        if not 0.1 <= scale <= 1.0:
            raise ValueError("scale must be in [0.1, 1.0]")
        self.swapper = swapper
        self.scale = scale
        self.detect_every = max(1, int(detect_every))
        self.smoother = LandmarkSmoother(alpha=smooth_alpha)
        self._frame_idx = 0

    def _detect(self, frame_rgb):
        inv = 1.0 / self.scale
        if self.scale != 1.0:
            small = cv2.resize(frame_rgb, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame_rgb

        faces = self.swapper.detector.detect_faces(small)
        detections = []
        for f in faces:
            landmarks = self.swapper.detector.get_landmarks(small, f)
            rect = dlib.rectangle(
                int(f.left() * inv), int(f.top() * inv),
                int(f.right() * inv), int(f.bottom() * inv),
            )
            detections.append((rect, landmarks * inv))
        return detections

    def process(self, frame_rgb):
        if self._frame_idx % self.detect_every == 0:
            self.smoother.update(self._detect(frame_rgb))
        self._frame_idx += 1

        faces = self.smoother.current()
        if not faces:
            return frame_rgb

        try:
            return self.swapper.swap_face(frame_rgb, fast=True, faces_override=faces)
        except Exception:
            return frame_rgb


def _mux_audio(silent_path, source_path, output_path):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        return False
    try:
        subprocess.run(
            [ffmpeg, '-y', '-i', str(silent_path), '-i', str(source_path),
             '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'copy', '-c:a', 'aac',
             '-shortest', str(output_path)],
            check=True, capture_output=True, timeout=600,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def process_video(swapper, input_path, output_path, scale=0.6, detect_every=3,
                   smooth_alpha=0.65, keep_audio=True, progress_cb=None):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    engine = FrameSwapEngine(swapper, scale=scale, detect_every=detect_every,
                             smooth_alpha=smooth_alpha)

    want_audio = keep_audio
    write_path = output_path
    tmp_silent = None
    if want_audio:
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp.close()
        tmp_silent = Path(tmp.name)
        write_path = tmp_silent

    writer = cv2.VideoWriter(str(write_path), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Could not write video: {write_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = engine.process(frame_rgb)
            writer.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            frame_idx += 1
            if progress_cb:
                progress_cb(frame_idx, total)
    finally:
        cap.release()
        writer.release()

    if want_audio:
        if _mux_audio(tmp_silent, input_path, output_path):
            tmp_silent.unlink(missing_ok=True)
        else:
            tmp_silent.replace(output_path)

    return output_path
