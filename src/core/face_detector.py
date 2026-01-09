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
        faces = self.detector(gray)
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

