from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.core.face_swapper import FaceSequence
import cv2
import dlib
import numpy as np

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def is_video_file(path):
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _shift_rect(rect, dx, dy):
    left = int(round(rect.left() + dx))
    top = int(round(rect.top() + dy))
    right = int(round(rect.right() + dx))
    bottom = int(round(rect.bottom() + dy))
    if right <= left:
        right = left + 1
    if bottom <= top:
        bottom = top + 1
    return dlib.rectangle(left, top, right, bottom)


class LandmarkSmoother:
    def __init__(self, responsiveness=0.85, max_misses=0, fps=30.0):
        self.max_misses = max(0, int(max_misses))
        self.gain = 0.7 + 0.45 * float(np.clip(responsiveness, 0.0, 1.0))
        self.tracks = {}
        self._next_id = 0
        self.dt = 1.0 / 30.0
        self.set_fps(fps)

    def set_fps(self, fps):
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = 30.0
        if not 1.0 <= fps <= 240.0:
            fps = 30.0
        self.dt = 1.0 / fps

    def set_step(self, seconds):
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return
        if seconds > 0:
            self.dt = min(max(seconds, 1.0 / 240.0), 0.5)

    @staticmethod
    def _center(rect):
        return np.array([(rect.left() + rect.right()) / 2.0,
                         (rect.top() + rect.bottom()) / 2.0])

    def _follow(self, track, landmarks):
        prev = track['landmarks']
        delta = landmarks - prev
        span = max(float(np.ptp(prev[:, 0])), float(np.ptp(landmarks[:, 0])), 1.0)
        unit = max(span * 0.025, 1.0)
        speed = np.linalg.norm(delta, axis=1) / unit
        alpha = np.clip(0.5 + speed * self.gain, 0.5, 1.0)[:, None]
        filtered = alpha * landmarks + (1.0 - alpha) * prev
        velocity = (filtered - prev) / self.dt
        return filtered, velocity

    def _coast_track(self, track):
        delta = track['velocity'] * self.dt
        track['landmarks'] = track['landmarks'] + delta
        shift = delta.mean(axis=0)
        track['rect'] = _shift_rect(track['rect'], float(shift[0]), float(shift[1]))

    def update(self, detections):
        used = set()
        output = []

        for rect, landmarks in detections:
            landmarks = np.asarray(landmarks, dtype=np.float64)
            if landmarks.ndim != 2 or landmarks.shape[0] < 3:
                continue
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
                self.tracks[best_id] = {
                    'rect': rect,
                    'landmarks': landmarks,
                    'velocity': np.zeros_like(landmarks),
                    'misses': 0,
                }
            else:
                track = self.tracks[best_id]
                filtered, velocity = self._follow(track, landmarks)
                track['landmarks'] = filtered
                track['velocity'] = velocity
                track['rect'] = rect
                track['misses'] = 0

            used.add(best_id)
            track = self.tracks[best_id]
            output.append((track['rect'], track['landmarks']))

        for tid in list(self.tracks):
            if tid in used:
                continue
            track = self.tracks[tid]
            track['misses'] += 1
            if track['misses'] > self.max_misses:
                del self.tracks[tid]
                continue
            self._coast_track(track)
            output.append((track['rect'], track['landmarks']))

        return output

    def coast(self):
        output = []
        for track in self.tracks.values():
            self._coast_track(track)
            output.append((track['rect'], track['landmarks']))
        return output


def _scale_tracked_faces(faces, scale, width, height):
    scaled = []
    for rect, landmarks in faces:
        landmarks = np.asarray(landmarks, dtype=np.float64) * scale
        left = min(max(int(round(rect.left() * scale)), 0), width - 1)
        top = min(max(int(round(rect.top() * scale)), 0), height - 1)
        right = min(max(int(round(rect.right() * scale)), left + 1), width)
        bottom = min(max(int(round(rect.bottom() * scale)), top + 1), height)
        scaled.append((dlib.rectangle(left, top, right, bottom), landmarks))
    return scaled


class FrameSwapEngine:

    def __init__(self, swapper, scale=0.6, detect_every=1, smooth_alpha=0.85,
                 max_edge=1280, fps=30.0):
        if not 0.1 <= scale <= 1.0:
            raise ValueError("scale must be in [0.1, 1.0]")
        self.swapper = swapper
        self.scale = scale
        self.detect_every = max(1, int(detect_every))
        self.max_edge = max_edge if max_edge and max_edge > 0 else None
        self.smoother = LandmarkSmoother(responsiveness=smooth_alpha, fps=fps)
        self._frame_idx = 0

    def set_step(self, seconds):
        self.smoother.set_step(seconds)

    def _detection_scale(self, frame_rgb):
        height, width = frame_rgb.shape[:2]
        long_edge = max(height, width)
        if long_edge <= 0:
            return self.scale
        target_long = long_edge * self.scale
        if self.max_edge:
            target_long = min(target_long, self.max_edge * self.scale)
        return target_long / float(long_edge)

    def _detect(self, frame_rgb):
        detect_scale = self._detection_scale(frame_rgb)
        inv = 1.0 / detect_scale if detect_scale else 1.0
        if detect_scale != 1.0:
            small = cv2.resize(frame_rgb, None, fx=detect_scale, fy=detect_scale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame_rgb

        faces = self.swapper.detector.detect_faces(small, fast=True)
        detections = []
        for f in faces:
            landmarks = self.swapper.detector.get_landmarks(small, f)
            rect = dlib.rectangle(
                int(f.left() * inv), int(f.top() * inv),
                int(f.right() * inv), int(f.bottom() * inv),
            )
            detections.append((rect, landmarks * inv))
        return detections

    def detect(self, frame_rgb):
        detect_now = self._frame_idx % self.detect_every == 0
        self._frame_idx += 1
        if not detect_now:
            return self.smoother.coast()
        try:
            detections = self._detect(frame_rgb)
        except Exception:
            detections = []
        return self.smoother.update(detections)

    def prepare_work(self, frame_rgb, faces):
        height, width = frame_rgb.shape[:2]
        output_size = (width, height)
        if not faces or not self.max_edge or max(height, width) <= self.max_edge:
            return frame_rgb, faces, output_size
        scale = self.max_edge / float(max(height, width))
        work_width = max(1, int(round(width * scale)))
        work_height = max(1, int(round(height * scale)))
        small = cv2.resize(frame_rgb, (work_width, work_height), interpolation=cv2.INTER_AREA)
        return small, _scale_tracked_faces(faces, scale, work_width, work_height), output_size

    def process_faces(self, frame_rgb, faces, output_size=None):
        if not faces:
            if output_size and (frame_rgb.shape[1], frame_rgb.shape[0]) != output_size:
                return cv2.resize(frame_rgb, output_size, interpolation=cv2.INTER_LINEAR)
            return frame_rgb

        try:
            swapped = self.swapper.swap_face(frame_rgb, fast=True, faces_override=faces)
        except Exception:
            swapped = frame_rgb
        if output_size and (swapped.shape[1], swapped.shape[0]) != output_size:
            swapped = cv2.resize(swapped, output_size, interpolation=cv2.INTER_LINEAR)
        return swapped

    def process(self, frame_rgb):
        faces = self.detect(frame_rgb)
        work, faces, output_size = self.prepare_work(frame_rgb, faces)
        return self.process_faces(work, faces, output_size)


def _normalize_rotation(angle):
    try:
        quarter = int(round(float(angle))) % 360
    except (TypeError, ValueError):
        return 0
    return quarter if quarter in (90, 180, 270) else 0


def _parse_rotation_text(text):
    tag = re.search(r"(?m)^\s*rotate\s*:\s*(-?\d+(?:\.\d+)?)", text)
    if tag:
        return _normalize_rotation(tag.group(1))
    matrix = re.search(r"rotation of ([-\d.]+) degrees", text)
    if matrix:
        return _normalize_rotation(-float(matrix.group(1)))
    return 0


def _probe_rotation(path):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        return 0
    try:
        result = subprocess.run(
            [ffmpeg, '-i', str(path), '-hide_banner'],
            capture_output=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return 0
    text = (result.stderr or b'').decode('utf-8', 'replace')
    return _parse_rotation_text(text)


def _media_has_audio(path):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        return None
    try:
        result = subprocess.run(
            [ffmpeg, '-i', str(path), '-hide_banner'],
            capture_output=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    text = (result.stderr or b'').decode('utf-8', 'replace')
    return 'Audio:' in text


def _display_rotation(cap, path):
    auto = getattr(cv2, 'CAP_PROP_ORIENTATION_AUTO', None)
    if auto is not None:
        cap.set(auto, 0)
    meta = getattr(cv2, 'CAP_PROP_ORIENTATION_META', None)
    angle = 0
    if meta is not None:
        try:
            angle = _normalize_rotation(cap.get(meta))
        except Exception:
            angle = 0
    if angle == 0:
        angle = _probe_rotation(path)
    return angle


def _rotate_frame(frame, angle):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _transcode_h264(silent_path, source_path, output_path, keep_audio):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        return False
    command = [ffmpeg, '-y', '-i', str(silent_path)]
    if keep_audio:
        command += ['-i', str(source_path), '-map', '0:v:0', '-map', '1:a:0?',
                    '-c:a', 'aac', '-shortest']
    else:
        command += ['-map', '0:v:0', '-an']
    command += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'veryfast',
                '-crf', '20', '-movflags', '+faststart', str(output_path)]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=600)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def _open_writer(path, fps, size):
    for fourcc in ('avc1', 'H264', 'X264', 'mp4v'):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise ValueError(f"Could not write video: {path}")


def _parse_rate(text):
    text = (text or '').strip()
    if not text or text == '0/0':
        return 0.0
    try:
        if '/' in text:
            num, den = text.split('/', 1)
            den_f = float(den)
            if den_f == 0:
                return 0.0
            return float(num) / den_f
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _sane_fps(fps):
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        return None
    if 1.0 < fps <= 240.0:
        return fps
    return None


def _probe_fps(path):
    ffprobe = shutil.which('ffprobe')
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            [ffprobe, '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=avg_frame_rate',
             '-of', 'default=nw=1:nk=1', str(path)],
            capture_output=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return _sane_fps(_parse_rate(result.stdout.decode('utf-8', 'replace')))


def _resolve_fps(path, opencv_fps):
    probed = _probe_fps(path)
    if probed:
        return probed
    return _sane_fps(opencv_fps) or 30.0


class _FfmpegPipeWriter:
    def __init__(self, path, fps, size, audio_source):
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            raise RuntimeError('ffmpeg is not installed')
        width, height = size
        self.size = (width, height)
        command = [
            ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', f'{float(fps):.6f}',
            '-i', 'pipe:0',
        ]
        if audio_source is not None:
            command += ['-i', str(audio_source), '-map', '0:v:0', '-map', '1:a:0?',
                        '-c:a', 'aac', '-shortest']
        else:
            command += ['-map', '0:v:0', '-an']
        command += [
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'veryfast',
            '-crf', '18', '-movflags', '+faststart', str(path),
        ]
        self.closed = False
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._err,
        )
        if self.proc.poll() not in (None, 0):
            self._err.seek(0)
            detail = self._err.read().decode('utf-8', 'replace')[-400:]
            self._err.close()
            raise RuntimeError(detail.strip() or 'ffmpeg failed to start')

    def write(self, bgr):
        width, height = self.size
        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        if bgr.dtype != np.uint8:
            bgr = np.clip(bgr, 0, 255).astype(np.uint8)
        if not bgr.flags['C_CONTIGUOUS']:
            bgr = np.ascontiguousarray(bgr)
        try:
            self.proc.stdin.write(bgr.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError('Video encoder stopped') from exc

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        try:
            code = self.proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = 1
        self._err.seek(0)
        detail = self._err.read().decode('utf-8', 'replace')[-400:]
        self._err.close()
        if code != 0:
            raise RuntimeError(detail.strip() or 'Video encoder failed')


def _fallback_rgb(frame_bgr, width, height):
    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] != width or rgb.shape[0] != height:
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        return rgb
    except Exception:
        return np.zeros((height, width, 3), dtype=np.uint8)


def process_video(swapper, input_path, output_path, scale=0.6, detect_every=1,
                   smooth_alpha=0.85, keep_audio=True, progress_cb=None,
                   workers=5, max_edge=1280, media_info=None):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    rotation = _display_rotation(cap, input_path)
    fps = _resolve_fps(input_path, cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if rotation in (90, 270):
        width, height = height, width
    width -= width % 2
    height -= height % 2
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video has invalid dimensions")

    sequence = FaceSequence(swapper)

    info = {"audio_dropped": False, "audio_note": None, "rotation": rotation}
    has_audio = _media_has_audio(input_path) if keep_audio else False
    tmp_silent = None
    pipe = None
    cv_writer = None
    try:
        pipe = _FfmpegPipeWriter(
            output_path, fps, (width, height),
            input_path if keep_audio and has_audio is not False else None,
        )
    except Exception:
        pipe = None

    if pipe is None:
        if shutil.which('ffmpeg'):
            tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            tmp.close()
            tmp_silent = Path(tmp.name)
            write_path = tmp_silent
            cv_writer = cv2.VideoWriter(
                str(write_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
            )
        else:
            write_path = output_path
            try:
                cv_writer = _open_writer(write_path, fps, (width, height))
            except ValueError:
                cap.release()
                raise
            if keep_audio and has_audio is not False:
                info["audio_dropped"] = True
                info["audio_note"] = "Audio was not copied because ffmpeg is not installed."
        if cv_writer is None or not cv_writer.isOpened():
            cap.release()
            if tmp_silent is not None:
                tmp_silent.unlink(missing_ok=True)
            raise ValueError(f"Could not write video: {output_path}")

    workers = max(1, int(workers))
    pending = deque()
    frame_idx = 0

    def _encode(rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        if pipe is not None:
            pipe.write(bgr)
        else:
            cv_writer.write(bgr)

    def write_next():
        nonlocal frame_idx
        future, fallback = pending.popleft()
        try:
            result = future.result()
            if result is None or result.size == 0:
                result = fallback
            _encode(result)
        except Exception:
            _encode(fallback)
        frame_idx += 1
        if progress_cb:
            progress_cb(frame_idx, total)

    try:
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                while True:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    frame_rgb = _fallback_rgb(
                        _rotate_frame(frame_bgr, rotation) if rotation else frame_bgr,
                        width, height,
                    )
                    try:
                        painted = sequence.apply(frame_rgb)
                    except Exception:
                        painted = frame_rgb
                    future = Future()
                    future.set_result(painted)
                    pending.append((future, frame_rgb))
                    if len(pending) >= workers * 2:
                        write_next()

                while pending:
                    write_next()
            if frame_idx == 0:
                raise ValueError("Video contains no readable frames")
        finally:
            cap.release()
            if cv_writer is not None:
                cv_writer.release()

        if pipe is not None:
            pipe.close()
            if keep_audio and has_audio and not _media_has_audio(output_path):
                info["audio_dropped"] = True
                info["audio_note"] = "Audio was not copied."
        elif tmp_silent is not None:
            copied_audio = bool(keep_audio and has_audio)
            if _transcode_h264(tmp_silent, input_path, output_path, copied_audio):
                if copied_audio and not _media_has_audio(output_path):
                    info["audio_dropped"] = True
                    info["audio_note"] = "Audio was not copied."
            else:
                tmp_silent.replace(output_path)
                if copied_audio:
                    info["audio_dropped"] = True
                    info["audio_note"] = "Audio was not copied because encoding failed."
    finally:
        if pipe is not None and not pipe.closed and pipe.proc.poll() is None:
            try:
                pipe.close()
            except Exception:
                pipe.proc.kill()
        if tmp_silent is not None:
            tmp_silent.unlink(missing_ok=True)

    if media_info is not None:
        media_info.update(info)
    return output_path
