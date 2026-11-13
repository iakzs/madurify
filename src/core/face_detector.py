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
        yunet_path = Path(get_models_path("face_detection_yunet_2023mar.onnx"))
        self._yunet_path = yunet_path if yunet_path.exists() else None
        self._yunet = None
        self._yunet_size = None

    def detect_faces(self, image, fast=False):
        gray = self._to_grayscale(image)
        h, w = gray.shape

        faces = self.detector(gray, 0)
        if len(faces) > 0:
            return faces

        # The frontal detector misses faces that are small or looking down.
        # YuNet still returns a box, and the landmark model fits that box.
        yunet_faces = self._detect_yunet(image)
        if len(yunet_faces) > 0 or fast:
            return yunet_faces

        for upsample in (1, 2):
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

    def _yunet_rows(self, image):
        if self._yunet_path is None or not hasattr(cv2, "FaceDetectorYN"):
            return None
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        if width < 20 or height < 20:
            return None
        try:
            if self._yunet is None:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(self._yunet_path), "", (width, height), 0.6, 0.3, 5000,
                )
                self._yunet_size = (width, height)
            elif self._yunet_size != (width, height):
                self._yunet.setInputSize((width, height))
                self._yunet_size = (width, height)
            _, found = self._yunet.detect(bgr)
        except cv2.error:
            return None
        return found

    def landmarks_match(self, image, landmarks):
        found = self._yunet_rows(image)
        if found is None or len(found) == 0:
            return False
        lm = np.asarray(landmarks, dtype=np.float64)
        if lm.shape[0] < 68:
            return False
        dlib_eyes = np.stack([lm[36:42].mean(axis=0), lm[42:48].mean(axis=0)])
        iod = float(np.linalg.norm(dlib_eyes[0] - dlib_eyes[1]))
        if iod < 8.0:
            return False
        nose = lm[30]
        limit = iod * 0.32
        for row in found:
            if len(row) < 14:
                continue
            keypoints = row[4:14].reshape(5, 2)
            right_eye, left_eye, yunet_nose = keypoints[0], keypoints[1], keypoints[2]
            eye_error = min(
                np.linalg.norm(dlib_eyes[0] - left_eye) + np.linalg.norm(dlib_eyes[1] - right_eye),
                np.linalg.norm(dlib_eyes[0] - right_eye) + np.linalg.norm(dlib_eyes[1] - left_eye),
            ) / 2.0
            if eye_error < limit and np.linalg.norm(nose - yunet_nose) < limit:
                return True
        return False

    def _detect_yunet(self, image):
        found = self._yunet_rows(image)
        if found is None:
            return []
        height, width = image.shape[:2]

        rects = []
        for row in found:
            x, y, box_w, box_h = (float(v) for v in row[:4])
            if box_w < 12 or box_h < 12 or len(row) < 14:
                continue
            right_eye, left_eye, _nose, right_mouth, left_mouth = row[4:14].reshape(5, 2)
            eye_gap = float(np.linalg.norm(left_eye - right_eye))
            if eye_gap < 8.0:
                continue
            # Eyes or mouth outside the picture means the face is cut off.
            if min(right_eye[0], left_eye[0], right_mouth[0], left_mouth[0]) < 1:
                continue
            if max(right_eye[0], left_eye[0], right_mouth[0], left_mouth[0]) > width - 2:
                continue
            if min(right_eye[1], left_eye[1]) < 1:
                continue
            left = min(right_eye[0], left_eye[0], right_mouth[0], left_mouth[0]) - eye_gap * 0.55
            right = max(right_eye[0], left_eye[0], right_mouth[0], left_mouth[0]) + eye_gap * 0.55
            top = min(right_eye[1], left_eye[1]) - eye_gap * 0.95
            bottom = max(right_mouth[1], left_mouth[1]) + eye_gap * 1.15
            left = int(max(0, min(round(left), width - 2)))
            top = int(max(0, min(round(top), height - 2)))
            right = int(max(left + 1, min(round(right), width - 1)))
            bottom = int(max(top + 1, min(round(bottom), height - 1)))
            rects.append(dlib.rectangle(left, top, right, bottom))
        return rects

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
