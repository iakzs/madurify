import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from src.core.video import FrameSwapEngine


class _LatestFrameGrabber:

    def __init__(self, cap):
        self.cap = cap
        self._lock = threading.Lock()
        self._latest = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._latest = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)


def run_camera(swapper, cam_index=0, width=960, height=540, scale=0.6,
               detect_every=3, smooth_alpha=0.65, mirror=True,
               snapshot_dir="."):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera {cam_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    engine = FrameSwapEngine(swapper, scale=scale, detect_every=detect_every,
                             smooth_alpha=smooth_alpha)
    grabber = _LatestFrameGrabber(cap)
    snapshot_dir = Path(snapshot_dir)

    print("Camera started. Press 'q' to quit, 's' for snapshot, 'f' to toggle mirror.")

    fps, frames, t0 = 0.0, 0, time.time()
    try:
        while True:
            frame_bgr = grabber.read()
            if frame_bgr is None:
                time.sleep(0.005)
                continue

            if mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result_rgb = engine.process(frame_rgb)
            display = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

            frames += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps, frames, t0 = frames / elapsed, 0, time.time()

            cv2.putText(display, f"{fps:.1f} FPS", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("madurify cam", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                name = f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(snapshot_dir / name), display)
                print(f"Snapshot saved: {snapshot_dir / name}")
            elif key == ord('f'):
                mirror = not mirror
    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop()
        cap.release()
        cv2.destroyAllWindows()
