import dlib
import cv2
import numpy as np
from pathlib import Path
from .paths import get_models_path


class FaceDetector:
    def __init__(self, predictor_path=None):
        self.detector = dlib.get_frontal_face_detector()

        if predictor_path is None:
            predictor_path = get_models_path("shape_predictor_68_face_landmarks.dat")

        if not Path(predictor_path).exists():
            raise FileNotFoundError(
                f"Landmark predictor not found at {predictor_path}. "
                "Please download shape_predictor_68_face_landmarks.dat from "
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )

        self.predictor = dlib.shape_predictor(str(predictor_path))

    def detect_faces(self, image):
        gray = self._to_grayscale(image)
        h, w = gray.shape

        for upsample in [0, 1, 2]:
            faces = self.detector(gray, upsample)
            if len(faces) > 0:
                return faces

        rects, scores, _ = self.detector.run(gray, 0, -0.5)
        if len(rects) > 0:
            return [rects[np.argmax(scores)]]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)

        for upsample in [0, 1]:
            faces = self.detector(equalized, upsample)
            if len(faces) > 0:
                return faces

        rects, scores, _ = self.detector.run(equalized, 0, -0.5)
        if len(rects) > 0:
            return [rects[np.argmax(scores)]]

        all_detections = []
        for scale in [0.75, 0.5, 0.33, 0.25]:
            small_w, small_h = int(w * scale), int(h * scale)
            if small_w < 100 or small_h < 100:
                continue
            small = cv2.resize(equalized, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            rects, scores, _ = self.detector.run(small, 0, -0.5)
            for rect, score in zip(rects, scores):
                all_detections.append((
                    score,
                    dlib.rectangle(
                        int(rect.left() / scale),
                        int(rect.top() / scale),
                        int(rect.right() / scale),
                        int(rect.bottom() / scale),
                    )
                ))

        if all_detections:
            all_detections.sort(key=lambda x: x[0], reverse=True)
            return [all_detections[0][1]]

        if max(h, w) < 500:
            scale = min(500.0 / max(h, w), 3.0)
            big = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for upsample in [0, 1]:
                faces = self.detector(big, upsample)
                if len(faces) > 0:
                    return [
                        dlib.rectangle(
                            int(f.left() / scale),
                            int(f.top() / scale),
                            int(f.right() / scale),
                            int(f.bottom() / scale),
                        ) for f in faces
                    ]

        return faces

    def get_landmarks(self, image, face):
        gray = self._to_grayscale(image)
        landmarks = self.predictor(gray, face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        return points

    def _to_grayscale(self, image):
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        return image
