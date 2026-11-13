import cv2
import numpy as np
from pathlib import Path
from .face_detector import FaceDetector
from .image_utils import load_image, save_image


def _shape_anchors(pts):
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    x0 -= bw * 0.2
    y0 -= bh * 0.2
    x1 += bw * 0.2
    y1 += bh * 0.2
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return np.array([
        [x0, y0], [mx, y0], [x1, y0],
        [x0, my],           [x1, my],
        [x0, y1], [mx, y1], [x1, y1],
    ], dtype=np.float32)


def _face_roi(points, width, height, margin=0.55):
    x0, y0 = float(points[:, 0].min()), float(points[:, 1].min())
    x1, y1 = float(points[:, 0].max()), float(points[:, 1].max())
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    left = max(0, int(np.floor(x0 - bw * margin)))
    top = max(0, int(np.floor(y0 - bh * margin)))
    right = min(width, int(np.ceil(x1 + bw * margin)))
    bottom = min(height, int(np.ceil(y1 + bh * margin)))
    return left, top, right, bottom


class FaceSwapper:
    RIGID_IDX = list(range(0, 17)) + list(range(27, 36)) + [39, 42]
    RIGID_MEAN_IDX = list(range(0, 17)) + list(range(27, 36))
    FACE_FEATURE_IDX = (set(range(17, 27)) | set(range(36, 39)) |
                        set(range(40, 42)) | set(range(43, 68)))

    def __init__(self, maduro_face_paths, predictor_path=None, debug=False):
        if isinstance(maduro_face_paths, str):
            maduro_face_paths = [maduro_face_paths]

        self.debug = debug
        self.maduro_faces = []
        self.maduro_landmarks_list = []
        self.detector = FaceDetector(predictor_path)

        for path in maduro_face_paths:
            face_img = load_image(path)
            faces = self.detector.detect_faces(face_img)
            if len(faces) > 0:
                landmarks = self.detector.get_landmarks(face_img, faces[0])
                self.maduro_faces.append(face_img)
                self.maduro_landmarks_list.append(landmarks)

        if len(self.maduro_faces) == 0:
            raise ValueError("No valid faces found in Maduro face images")

    def swap_face(self, target_image, debug_path=None, fast=False, faces_override=None):
        image = np.asarray(target_image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        sequence = FaceSequence(self)
        result = sequence.apply(image)
        if not sequence.tracks:
            raise ValueError("No face detected in target image")
        return result

    def _select_best_source(self, target_landmarks):
        if len(self.maduro_faces) == 1:
            return 0

        target_left_eye = np.mean(target_landmarks[36:42], axis=0)
        target_right_eye = np.mean(target_landmarks[42:48], axis=0)
        target_nose = target_landmarks[30]
        target_nose_bridge = target_landmarks[27]
        target_mouth_left = target_landmarks[48]
        target_mouth_right = target_landmarks[54]

        target_eye_center = (target_left_eye + target_right_eye) / 2
        t_w = np.linalg.norm(target_right_eye - target_left_eye)
        target_nose_offset = (target_nose[0] - target_eye_center[0]) / max(t_w, 1)
        target_nose_length = np.linalg.norm(target_nose - target_nose_bridge) / max(t_w, 1)
        target_mouth_width = np.linalg.norm(target_mouth_right - target_mouth_left) / max(t_w, 1)
        target_jaw_width = np.linalg.norm(target_landmarks[0] - target_landmarks[16]) / max(t_w, 1)

        best_idx = 0
        best_score = float('inf')

        for idx, src_lm in enumerate(self.maduro_landmarks_list):
            src_left_eye = np.mean(src_lm[36:42], axis=0)
            src_right_eye = np.mean(src_lm[42:48], axis=0)
            src_nose = src_lm[30]
            src_nose_bridge = src_lm[27]
            src_mouth_left = src_lm[48]
            src_mouth_right = src_lm[54]

            src_eye_center = (src_left_eye + src_right_eye) / 2
            s_w = np.linalg.norm(src_right_eye - src_left_eye)
            src_nose_offset = (src_nose[0] - src_eye_center[0]) / max(s_w, 1)
            src_nose_length = np.linalg.norm(src_nose - src_nose_bridge) / max(s_w, 1)
            src_mouth_width = np.linalg.norm(src_mouth_right - src_mouth_left) / max(s_w, 1)
            src_jaw_width = np.linalg.norm(src_lm[0] - src_lm[16]) / max(s_w, 1)

            score = (abs(target_nose_offset - src_nose_offset) * 2.0 +
                     abs(target_nose_length - src_nose_length) * 1.0 +
                     abs(target_mouth_width - src_mouth_width) * 1.0 +
                     abs(target_jaw_width - src_jaw_width) * 1.0 +
                     (4.0 if self._source_has_mic(idx) else 0.0))

            if score < best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _source_has_mic(self, index):
        image = self.maduro_faces[index]
        landmarks = np.asarray(self.maduro_landmarks_list[index], dtype=np.float64)
        height, width = image.shape[:2]
        lip = landmarks[57]
        chin = landmarks[8]
        probe = lip + (chin - lip) * 0.45
        cheek = landmarks[31]
        px = int(np.clip(probe[0], 0, width - 1))
        py = int(np.clip(probe[1], 0, height - 1))
        cx = int(np.clip(cheek[0], 0, width - 1))
        cy = int(np.clip(cheek[1], 0, height - 1))
        probe_l = float(image[py, px].astype(np.int16).mean())
        cheek_l = float(image[cy, cx].astype(np.int16).mean())
        return probe_l < cheek_l * 0.62

    def _compute_eye_similarity(self, src_points, dst_points):
        src_left = np.mean(src_points[36:42], axis=0)
        src_right = np.mean(src_points[42:48], axis=0)
        dst_left = np.mean(dst_points[36:42], axis=0)
        dst_right = np.mean(dst_points[42:48], axis=0)

        src_center = (src_left + src_right) / 2
        dst_center = (dst_left + dst_right) / 2

        src_dist = np.linalg.norm(src_right - src_left)
        dst_dist = np.linalg.norm(dst_right - dst_left)
        scale = dst_dist / max(src_dist, 1)

        src_angle = np.arctan2(src_right[1] - src_left[1],
                               src_right[0] - src_left[0])
        dst_angle = np.arctan2(dst_right[1] - dst_left[1],
                               dst_right[0] - dst_left[0])
        rotation = dst_angle - src_angle

        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        M = np.zeros((2, 3), dtype=np.float64)
        M[0, 0] = scale * cos_r
        M[0, 1] = -scale * sin_r
        M[1, 0] = scale * sin_r
        M[1, 1] = scale * cos_r
        M[0, 2] = dst_center[0] - scale * (cos_r * src_center[0] - sin_r * src_center[1])
        M[1, 2] = dst_center[1] - scale * (sin_r * src_center[0] + cos_r * src_center[1])

        return M

    @staticmethod
    def _shape_strength(landmarks, image_shape):
        lm = np.asarray(landmarks, dtype=np.float64)
        if lm.shape[0] < 68:
            return 0.0
        height, width = image_shape[:2]
        left_eye = lm[36:42].mean(axis=0)
        right_eye = lm[42:48].mean(axis=0)
        iod = float(np.linalg.norm(right_eye - left_eye))
        if iod < 12.0:
            return 0.0
        eye_vec = right_eye - left_eye
        nose_t = float(np.dot(lm[30] - left_eye, eye_vec) / (np.dot(eye_vec, eye_vec) + 1e-6))
        face_height = float(lm[8, 1] - lm[27, 1]) / iod
        margin = min(
            float(lm[:, 0].min()), float(lm[:, 1].min()),
            width - 1 - float(lm[:, 0].max()), height - 1 - float(lm[:, 1].max()),
        )
        roll = abs(float(np.arctan2(eye_vec[1], eye_vec[0])))
        anatomy = left_eye[1] < lm[30, 1] < lm[51, 1] < lm[8, 1]
        # Points touching the frame edge belong to a cropped face. Warping
        # them smears the border, so those faces are left alone.
        if not anatomy or roll > np.radians(35) or margin < 12 or not 0.18 <= nose_t <= 0.90:
            return 0.0
        if face_height < 0.85 or face_height > 2.35:
            return 0.0
        full = (
            0.34 <= nose_t <= 0.80
            and 0.9 <= face_height <= 2.05
            and margin >= 10
            and roll <= np.radians(28)
        )
        return 1.0 if full else 0.45

    def _compose_shape(self, target_landmarks, source_landmarks, strength=1.0):
        target = np.asarray(target_landmarks, dtype=np.float64)
        source = np.asarray(source_landmarks, dtype=np.float64)
        if target.shape[0] < 68 or source.shape[0] < 68:
            return target.copy()

        left_eye = target[36:42].mean(axis=0)
        right_eye = target[42:48].mean(axis=0)
        iod = float(np.linalg.norm(right_eye - left_eye))
        if iod < 8.0:
            return target.copy()

        aligned = self._transform_points(source, self._compute_eye_similarity(source, target))
        amount = float(np.clip(strength, 0.0, 1.0))
        cap = iod * 0.5
        dest = target.copy()

        jaw_w = np.array([
            0.35, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.7, 0.35,
        ]) * amount
        jaw_delta = aligned[0:17] - target[0:17]
        jaw_delta[:, 0] = np.clip(jaw_delta[:, 0], -cap, cap)
        jaw_delta[:, 1] = np.clip(jaw_delta[:, 1], -iod * 0.22, iod * 0.16)
        dest[0:17] = target[0:17] + jaw_w[:, None] * jaw_delta

        # Width is already close to the template, so the raw delta barely
        # shows. Force the cheeks outward and keep the shorter chin.
        center_x = float((left_eye[0] + right_eye[0]) * 0.5)
        flare = np.array([
            0.0, 0.06, 0.14, 0.22, 0.28, 0.22, 0.10, 0.0, 0.0,
            0.0, 0.10, 0.22, 0.28, 0.22, 0.14, 0.06, 0.0,
        ])
        for index in range(17):
            sign = -1.0 if target[index, 0] < center_x else 1.0
            outward = iod * flare[index] * amount
            moved = dest[index, 0] - target[index, 0]
            if sign * moved < outward:
                dest[index, 0] = target[index, 0] + sign * outward

        nose_delta = np.clip(aligned[27:36] - target[27:36], -cap * 0.7, cap * 0.7)
        dest[27:36] = target[27:36] + (0.85 * amount) * nose_delta
        dest[31, 0] = min(dest[31, 0], target[31, 0] - iod * 0.1 * amount)
        dest[35, 0] = max(dest[35, 0], target[35, 0] + iod * 0.1 * amount)

        chin_move = dest[8] - target[8]
        dest[48:68] = target[48:68] + chin_move * np.array([0.12, 0.55])
        return dest

    @staticmethod
    def _face_plate(image, landmarks):
        plate = np.array(image, copy=True)
        hull = cv2.convexHull(np.round(np.asarray(landmarks, dtype=np.float64)).astype(np.int32))
        mask = np.zeros(plate.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(mask, grow)
        pixels = plate[mask > 0]
        if len(pixels) == 0:
            return plate
        plate[mask == 0] = np.median(pixels, axis=0)
        return plate

    def _apply_identity(self, result, target_image, target_landmarks,
                        source_img, source_landmarks, strength):
        placed_on = self._tracked_features(target_image, target_landmarks)
        if placed_on is None:
            return False
        dst_left, dst_right, dst_mouth = placed_on
        src_left = np.asarray(source_landmarks[36:42], dtype=np.float64).mean(axis=0)
        src_right = np.asarray(source_landmarks[42:48], dtype=np.float64).mean(axis=0)
        if src_left[0] > src_right[0]:
            src_left, src_right = src_right, src_left
        matrix = self._eye_matrix(src_left, src_right, dst_left, dst_right)
        plate = self._face_plate(source_img, source_landmarks)
        height, width = target_image.shape[:2]
        warped = cv2.warpAffine(
            plate, matrix, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        mask = self._feature_ellipse(dst_left, dst_right, dst_mouth, (height, width), strength)
        if int(np.count_nonzero(mask)) < 40:
            return False
        matched = self._correct_colors(target_image, warped, mask, target_landmarks)
        mixed = np.clip(
            warped.astype(np.float32) * 0.7 + matched.astype(np.float32) * 0.3,
            0, 255,
        ).astype(np.uint8)
        blended = self._multi_band_blend(result, mixed, mask, levels=4)
        result[:] = np.clip(blended, 0, 255).astype(np.uint8)
        return True

    def _tracked_features(self, image, landmarks):
        found = self.detector._yunet_rows(image)
        if found is None or len(found) == 0:
            return None
        nose = np.asarray(landmarks[30], dtype=np.float64)
        best = None
        best_dist = 1e9
        for row in found:
            if len(row) < 14:
                continue
            points = row[4:14].reshape(5, 2).astype(np.float64)
            dist = float(np.linalg.norm(points[2] - nose))
            if dist < best_dist:
                best_dist = dist
                best = points
        if best is None:
            return None
        eyes = best[:2]
        iod = float(np.linalg.norm(eyes[0] - eyes[1]))
        if iod < 8 or best_dist > iod * 0.8:
            return None
        image_left = eyes[np.argmin(eyes[:, 0])]
        image_right = eyes[np.argmax(eyes[:, 0])]
        mouth = best[3:5].mean(axis=0)
        eye_center = (image_left + image_right) / 2.0
        reach = float(np.linalg.norm(mouth - eye_center)) / iod
        # A face looking hard down or up compresses this and the paste misses the head.
        if not 0.95 <= reach <= 1.45:
            return None
        return image_left, image_right, mouth

    @staticmethod
    def _eye_matrix(src_left, src_right, dst_left, dst_right):
        src_center = (src_left + src_right) / 2.0
        dst_center = (dst_left + dst_right) / 2.0
        src_dist = float(np.linalg.norm(src_right - src_left))
        dst_dist = float(np.linalg.norm(dst_right - dst_left))
        scale = dst_dist / max(src_dist, 1.0)
        rotation = (
            np.arctan2(dst_right[1] - dst_left[1], dst_right[0] - dst_left[0])
            - np.arctan2(src_right[1] - src_left[1], src_right[0] - src_left[0])
        )
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        matrix = np.zeros((2, 3), dtype=np.float64)
        matrix[0, 0] = scale * cos_r
        matrix[0, 1] = -scale * sin_r
        matrix[1, 0] = scale * sin_r
        matrix[1, 1] = scale * cos_r
        matrix[0, 2] = dst_center[0] - scale * (cos_r * src_center[0] - sin_r * src_center[1])
        matrix[1, 2] = dst_center[1] - scale * (sin_r * src_center[0] + cos_r * src_center[1])
        return matrix

    @staticmethod
    def _feature_ellipse(left_eye, right_eye, mouth, image_shape, strength):
        height, width = image_shape[:2]
        eye_center = (left_eye + right_eye) / 2.0
        iod = float(np.linalg.norm(right_eye - left_eye))
        down = mouth - eye_center
        reach = float(np.linalg.norm(down))
        if reach < iod * 0.3:
            down = np.array([0.0, iod])
            reach = iod
        center = eye_center + down * 0.55
        angle = float(np.degrees(np.arctan2(down[1], down[0])) - 90.0)
        axes = (max(int(iod * 1.05), 4), max(int(reach * 1.1), 4))
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (int(round(center[0])), int(round(center[1]))),
            axes, angle, 0, 360, 255, -1,
        )
        soft = cv2.GaussianBlur(mask, (0, 0), 0.6)
        core_size = max(3, int(iod * 0.2) | 1)
        core = cv2.erode(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_size, core_size)),
        )
        soft[core > 0] = 255
        return soft

    @staticmethod
    def _transform_points(points, matrix):
        pts = np.asarray(points, dtype=np.float64)
        out = np.empty_like(pts)
        out[:, 0] = matrix[0, 0] * pts[:, 0] + matrix[0, 1] * pts[:, 1] + matrix[0, 2]
        out[:, 1] = matrix[1, 0] * pts[:, 0] + matrix[1, 1] * pts[:, 1] + matrix[1, 2]
        return out

    @staticmethod
    def _forehead_points(landmarks):
        lm = np.asarray(landmarks, dtype=np.float64)
        left_eye = lm[36:42].mean(axis=0)
        right_eye = lm[42:48].mean(axis=0)
        eye_vec = right_eye - left_eye
        iod = float(np.linalg.norm(eye_vec)) + 1e-6
        up = np.array([-eye_vec[1], eye_vec[0]], dtype=np.float64)
        up /= float(np.linalg.norm(up)) + 1e-6
        if up[1] > 0:
            up = -up
        brow = lm[17:27].mean(axis=0)
        left = lm[17]
        right = lm[26]
        span = right - left
        return np.stack([
            brow + up * iod * 0.42,
            left + up * iod * 0.5 - span * 0.12,
            right + up * iod * 0.5 + span * 0.12,
            brow + up * iod * 0.95,
        ])

    @staticmethod
    def _outer_ring(landmarks, scale=1.12, count=18):
        lm = np.asarray(landmarks, dtype=np.float64)
        center = lm.mean(axis=0)
        rx = max(float(lm[:, 0].max() - lm[:, 0].min()) * 0.5 * scale, 4.0)
        ry = max(float(lm[:, 1].max() - lm[:, 1].min()) * 0.5 * scale, 4.0)
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        return np.stack([
            center[0] + np.cos(angles) * rx,
            center[1] + np.sin(angles) * ry,
        ], axis=1)

    def _control_points(self, src_landmarks, dst_landmarks, lock_ring):
        src = np.asarray(src_landmarks, dtype=np.float64)
        dst = np.asarray(dst_landmarks, dtype=np.float64)
        src_extra = self._forehead_points(src)
        dst_extra = self._forehead_points(dst)
        if lock_ring:
            ring = self._outer_ring(src)
            src_ring = ring
            dst_ring = ring
        else:
            src_ring = self._outer_ring(src)
            dst_ring = self._outer_ring(dst)
        return (
            np.vstack([src, src_extra, src_ring]).astype(np.float32),
            np.vstack([dst, dst_extra, dst_ring]).astype(np.float32),
        )

    def _remap_mesh(self, sample, src_landmarks, dst_landmarks, canvas=None, lock_ring=False):
        from scipy.spatial import Delaunay as ScipyDelaunay

        sample = np.asarray(sample)
        if sample.ndim != 3 or sample.shape[2] != 3:
            return None
        if canvas is None:
            canvas = sample
        if canvas.shape[2] != 3:
            return None

        src_pts, dst_pts = self._control_points(src_landmarks, dst_landmarks, lock_ring)
        out_h, out_w = canvas.shape[:2]
        margin = 2.0
        dst_pts[:, 0] = np.clip(dst_pts[:, 0], margin, out_w - 1 - margin)
        dst_pts[:, 1] = np.clip(dst_pts[:, 1], margin, out_h - 1 - margin)
        if lock_ring:
            src_pts[-18:] = dst_pts[-18:]

        left, top, right, bottom = _face_roi(dst_pts, out_w, out_h, margin=0.08)
        roi_w, roi_h = right - left, bottom - top
        if roi_w < 12 or roi_h < 12:
            return None

        local = dst_pts.copy()
        local[:, 0] -= left
        local[:, 1] -= top
        try:
            triangles = ScipyDelaunay(local).simplices
        except Exception:
            return None

        map_x = np.zeros((roi_h, roi_w), dtype=np.float32)
        map_y = np.zeros((roi_h, roi_w), dtype=np.float32)
        coverage = np.zeros((roi_h, roi_w), dtype=np.uint8)

        for simplex in triangles:
            src_tri = src_pts[simplex]
            dst_tri = local[simplex]
            src_area = ((src_tri[1, 0] - src_tri[0, 0]) * (src_tri[2, 1] - src_tri[0, 1]) -
                        (src_tri[1, 1] - src_tri[0, 1]) * (src_tri[2, 0] - src_tri[0, 0]))
            dst_area = ((dst_tri[1, 0] - dst_tri[0, 0]) * (dst_tri[2, 1] - dst_tri[0, 1]) -
                        (dst_tri[1, 1] - dst_tri[0, 1]) * (dst_tri[2, 0] - dst_tri[0, 0]))
            if abs(src_area) < 1.0 or abs(dst_area) < 1.0 or src_area * dst_area < 0:
                continue
            try:
                bx, by, bw, bh = cv2.boundingRect(np.ascontiguousarray(dst_tri))
            except cv2.error:
                continue
            x0, y0 = max(0, bx), max(0, by)
            x1, y1 = min(roi_w, bx + bw), min(roi_h, by + bh)
            tw, th = x1 - x0, y1 - y0
            if tw < 2 or th < 2:
                continue
            patch_tri = dst_tri - np.float32([[x0, y0], [x0, y0], [x0, y0]])
            patch = np.zeros((th, tw), dtype=np.uint8)
            cv2.fillConvexPoly(patch, np.round(patch_tri).astype(np.int32), 255)
            ys, xs = np.nonzero(patch)
            if len(xs) == 0:
                continue
            try:
                matrix = cv2.getAffineTransform(patch_tri, src_tri)
            except cv2.error:
                continue
            px = xs.astype(np.float32)
            py = ys.astype(np.float32)
            coverage[y0 + ys, x0 + xs] = 255
            map_x[y0 + ys, x0 + xs] = matrix[0, 0] * px + matrix[0, 1] * py + matrix[0, 2]
            map_y[y0 + ys, x0 + xs] = matrix[1, 0] * px + matrix[1, 1] * py + matrix[1, 2]

        if int(np.count_nonzero(coverage)) < 40:
            return None

        sampled = cv2.remap(
            sample, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        out = np.array(canvas, copy=True)
        roi = out[top:bottom, left:right]
        taken = coverage > 0
        roi[taken] = sampled[taken]
        out[top:bottom, left:right] = roi

        full = np.zeros(canvas.shape[:2], dtype=np.uint8)
        full[top:bottom, left:right] = coverage
        return out, full

    def _warp_shape_roi(self, source_img, source_points, target_img, target_points):
        from scipy.spatial import Delaunay as ScipyDelaunay

        height, width = target_img.shape[:2]
        source_points = np.asarray(source_points, dtype=np.float32)
        target_points = np.asarray(target_points, dtype=np.float32)
        left, top, right, bottom = _face_roi(target_points, width, height)
        roi_w, roi_h = right - left, bottom - top
        if roi_w < 12 or roi_h < 12:
            return None

        dst = target_points.copy()
        dst[:, 0] -= left
        dst[:, 1] -= top

        all_src = np.vstack([source_points, _shape_anchors(source_points)])
        all_dst = np.vstack([dst, _shape_anchors(dst)])
        try:
            triangles = ScipyDelaunay(all_dst).simplices
        except Exception:
            return None

        warped_sum = np.zeros((roi_h, roi_w, 3), dtype=np.float32)
        weight_sum = np.zeros((roi_h, roi_w), dtype=np.float32)

        for simplex in triangles:
            src_tri = all_src[simplex]
            dst_tri = all_dst[simplex]
            src_area = ((src_tri[1][0] - src_tri[0][0]) * (src_tri[2][1] - src_tri[0][1]) -
                        (src_tri[1][1] - src_tri[0][1]) * (src_tri[2][0] - src_tri[0][0]))
            dst_area = ((dst_tri[1][0] - dst_tri[0][0]) * (dst_tri[2][1] - dst_tri[0][1]) -
                        (dst_tri[1][1] - dst_tri[0][1]) * (dst_tri[2][0] - dst_tri[0][0]))
            if abs(src_area) < 1.0 or abs(dst_area) < 1.0:
                continue

            try:
                bx, by, bw, bh = cv2.boundingRect(np.ascontiguousarray(dst_tri))
            except cv2.error:
                continue
            x, y = max(0, bx), max(0, by)
            x1, y1 = min(roi_w, bx + bw), min(roi_h, by + bh)
            tw, th = x1 - x, y1 - y
            if tw < 2 or th < 2:
                continue

            dst_offset = dst_tri - np.float32([[x, y], [x, y], [x, y]])
            mask = np.zeros((th, tw), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.round(dst_offset).astype(np.int32), 255,
                               lineType=cv2.LINE_AA)
            try:
                patch = cv2.warpAffine(
                    source_img, cv2.getAffineTransform(src_tri, dst_offset), (tw, th),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )
            except cv2.error:
                continue
            weight = mask.astype(np.float32) / 255.0
            warped_sum[y:y + th, x:x + tw] += patch.astype(np.float32) * weight[:, :, None]
            weight_sum[y:y + th, x:x + tw] += weight

        if int(np.count_nonzero(weight_sum > 1e-5)) < 20:
            return None

        result = target_img.copy()
        roi = result[top:bottom, left:right].astype(np.float32)
        valid = weight_sum > 1e-5
        warped_sum[valid] /= weight_sum[valid, None]
        coverage = np.clip(weight_sum, 0.0, 1.0)
        roi[valid] = (warped_sum[valid] * coverage[valid, None] +
                      roi[valid] * (1.0 - coverage[valid, None]))
        result[top:bottom, left:right] = np.clip(roi, 0, 255).astype(np.uint8)
        return result

    def _warp_face(self, source_img, source_landmarks, target_img, target_landmarks, fast=False):
        h, w = target_img.shape[:2]

        source_points = source_landmarks.astype(np.float32)
        target_points = target_landmarks.astype(np.float32)

        M = self._compute_eye_similarity(source_points, target_points)
        if fast:
            warped = None
            try:
                warped = self._warp_shape_roi(
                    source_img, source_points, target_img, target_points
                )
            except Exception:
                warped = None
            if warped is None:
                warped = cv2.warpAffine(
                    source_img, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            return warped, M, target_points

        from scipy.spatial import Delaunay as ScipyDelaunay

        def _bbox_anchors(pts, expand=0.15):
            x0, y0 = pts[:, 0].min(), pts[:, 1].min()
            x1, y1 = pts[:, 0].max(), pts[:, 1].max()
            bw, bh = x1 - x0, y1 - y0
            x0 -= bw * expand;  y0 -= bh * expand
            x1 += bw * expand;  y1 += bh * expand
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            return np.array([
                [x0, y0], [mx, y0], [x1, y0],
                [x0, my],           [x1, my],
                [x0, y1], [mx, y1], [x1, y1],
            ], dtype=np.float32)

        src_anchors = _bbox_anchors(source_points)
        dst_anchors = _bbox_anchors(target_points)

        all_src = np.vstack([source_points, src_anchors])
        all_dst = np.vstack([target_points, dst_anchors])

        tri = ScipyDelaunay(all_dst)
        warped_sum = np.zeros((h, w, 3), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

        for simplex in tri.simplices:
            src_tri = all_src[simplex]
            dst_tri = all_dst[simplex]

            src_area = ((src_tri[1][0] - src_tri[0][0]) * (src_tri[2][1] - src_tri[0][1]) -
                        (src_tri[1][1] - src_tri[0][1]) * (src_tri[2][0] - src_tri[0][0]))
            dst_area = ((dst_tri[1][0] - dst_tri[0][0]) * (dst_tri[2][1] - dst_tri[0][1]) -
                        (dst_tri[1][1] - dst_tri[0][1]) * (dst_tri[2][0] - dst_tri[0][0]))
            if abs(src_area) < 1.0 or abs(dst_area) < 1.0:
                continue

            bx, by, bw, bh = cv2.boundingRect(dst_tri)
            x, y = max(0, bx), max(0, by)
            x1, y1 = min(w, bx + bw), min(h, by + bh)
            tw, th = x1 - x, y1 - y
            if tw < 2 or th < 2:
                continue

            dst_offset = dst_tri - np.float32([[x, y], [x, y], [x, y]])
            mask = np.zeros((th, tw), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.round(dst_offset).astype(np.int32), 255,
                               lineType=cv2.LINE_AA)

            tri_M = cv2.getAffineTransform(src_tri, dst_offset)
            patch = cv2.warpAffine(source_img, tri_M, (tw, th),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

            weight = mask.astype(np.float32) / 255.0
            warped_sum[y:y+th, x:x+tw] += patch.astype(np.float32) * weight[:, :, None]
            weight_sum[y:y+th, x:x+tw] += weight

        coverage = np.clip(weight_sum, 0.0, 1.0)
        valid = weight_sum > 1e-5
        result = target_img.astype(np.float32).copy()
        warped_sum[valid] /= weight_sum[valid, None]
        result[valid] = (warped_sum[valid] * coverage[valid, None] +
                         result[valid] * (1.0 - coverage[valid, None]))
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result, M, target_points

    def _create_face_mask(self, landmarks, img_shape, target_img=None):
        h, w = img_shape
        points = landmarks.astype(np.float32)

        jaw = points[0:17]
        left_brow = points[17:22]
        right_brow = points[22:27]
        left_eye = points[36:42]
        right_eye = points[42:48]

        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        brow_center = np.mean(np.vstack([left_brow, right_brow]), axis=0)
        chin = points[8]
        face_height = np.linalg.norm(chin - brow_center)
        face_width = np.linalg.norm(right_eye_center - left_eye_center)

        eye_angle = np.arctan2(right_eye_center[1] - left_eye_center[1],
                               right_eye_center[0] - left_eye_center[0])
        up_dx = -np.sin(eye_angle)
        up_dy = -np.cos(eye_angle)

        all_brow = np.vstack([left_brow, right_brow])
        brow_bottom_y = np.max(all_brow[:, 1])

        adjusted_jaw = jaw.copy()

        brow_left_outer = points[17]
        brow_right_outer = points[26]

        cheek_left = points[1]
        cheek_right = points[15]

        face_contour = [adjusted_jaw[0]]
        for i in range(1, 9):
            face_contour.append(adjusted_jaw[i])
        inner_cheek_left = points[2] + (points[1] - points[2]) * 0.3
        face_contour.append(inner_cheek_left)
        face_contour.append(cheek_left + (brow_left_outer - cheek_left) * 0.15)
        face_contour.append(brow_left_outer + np.array([-face_width * 0.05, 0]))

        up_projection = np.array([0, -face_height * 0.10])
        forehead_left = points[17] + up_projection
        forehead_right = points[26] + up_projection

        face_contour.append(forehead_left)
        face_contour.append(forehead_right)

        face_contour.append(brow_right_outer + np.array([face_width * 0.05, 0]))
        face_contour.append(cheek_right + (brow_right_outer - cheek_right) * 0.15)
        inner_cheek_right = points[14] + (points[15] - points[14]) * 0.3
        face_contour.append(inner_cheek_right)
        for i in range(15, 8, -1):
            face_contour.append(adjusted_jaw[i])
        face_contour.append(adjusted_jaw[16])

        face_contour = np.array(face_contour, dtype=np.int32)
        hull = cv2.convexHull(face_contour)

        hard_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(hard_mask, hull, 255)

        eye_and_brow_pts = points[17:27].tolist() + points[36:48].tolist()
        for pt in eye_and_brow_pts:
            cv2.circle(hard_mask, (int(pt[0]), int(pt[1])), int(face_width * 0.08), 255, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hard_mask = cv2.erode(hard_mask, kernel, iterations=2)

        blur_size = max(int(face_height * 0.07) | 1, 11)
        mask = cv2.GaussianBlur(hard_mask, (blur_size, blur_size), 0)

        if target_img is not None:
            gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            grad_mag = cv2.GaussianBlur(grad_mag, (5, 5), 0)
            grad_norm = np.clip(grad_mag / 30.0, 0, 1)

            boundary_dilate = cv2.dilate((hard_mask > 0).astype(np.uint8),
                                         np.ones((9, 9), np.uint8), iterations=1)
            boundary_erode = cv2.erode((hard_mask > 0).astype(np.uint8),
                                       np.ones((9, 9), np.uint8), iterations=1)
            boundary_zone = (boundary_dilate > 0) & (boundary_erode == 0)

            eye_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(eye_mask, landmarks[36:42].astype(np.int32), 255)
            cv2.fillConvexPoly(eye_mask, landmarks[42:48].astype(np.int32), 255)
            eye_mask = cv2.dilate(eye_mask, np.ones((15, 15), np.uint8), iterations=1)

            valid_penalty_zone = boundary_zone & (eye_mask == 0)

            edge_penalty = np.ones((h, w), dtype=np.float32)
            edge_penalty[valid_penalty_zone] = 1.0 - grad_norm[valid_penalty_zone] * 0.25

            mask = (mask.astype(np.float32) * edge_penalty)
            mask = np.clip(mask, 0, 255).astype(np.uint8)

        return mask

    def _correct_colors(self, target_img, warped_face, mask, landmarks=None):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return warped_face
        full_warped = warped_face
        h, w = mask.shape
        padding = 8
        x0, x1 = max(0, int(xs.min()) - padding), min(w, int(xs.max()) + padding + 1)
        y0, y1 = max(0, int(ys.min()) - padding), min(h, int(ys.max()) + padding + 1)
        target_img = target_img[y0:y1, x0:x1]
        warped_face = warped_face[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]
        mask_float = mask.astype(np.float32) / 255.0

        base_interior = mask_float > 0.9
        if np.sum(base_interior) < 100:
            return full_warped

        if landmarks is not None:
            local_landmarks = np.asarray(landmarks, dtype=np.float32).copy()
            local_landmarks[:, 0] -= x0
            local_landmarks[:, 1] -= y0
            feature_mask = np.zeros(mask.shape, dtype=np.uint8)
            eye_distance = np.linalg.norm(
                np.mean(local_landmarks[42:48], axis=0) -
                np.mean(local_landmarks[36:42], axis=0)
            )
            for eye_points in (local_landmarks[36:42], local_landmarks[42:48]):
                cv2.fillConvexPoly(feature_mask, cv2.convexHull(eye_points.astype(np.int32)), 255)
            mouth_points = local_landmarks[48:60].astype(np.int32)
            cv2.fillConvexPoly(feature_mask, cv2.convexHull(mouth_points), 255)
            feature_margin = max(3, int(eye_distance * 0.12) | 1)
            feature_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (feature_margin, feature_margin)
            )
            feature_mask = cv2.dilate(feature_mask, feature_kernel, iterations=1)
            base_interior &= feature_mask == 0
            if np.count_nonzero(base_interior) < 100:
                return full_warped

        warped_gray = cv2.cvtColor(warped_face, cv2.COLOR_RGB2GRAY)
        skin_pixels_mask = warped_gray > 65

        interior = base_interior & skin_pixels_mask

        if np.sum(interior) < 100:
            interior = base_interior

        target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        warped_lab = cv2.cvtColor(warped_face, cv2.COLOR_RGB2LAB).astype(np.float32)

        target_skin = target_lab[interior]
        warped_skin = warped_lab[interior]

        if len(target_skin) < 100 or len(warped_skin) < 100:
            return full_warped

        result_lab = warped_lab.copy()

        for c in range(3):
            w_ch = warped_skin[:, c]
            t_ch = target_skin[:, c]

            w_center, t_center = np.median(w_ch), np.median(t_ch)
            w_low, w_high = np.percentile(w_ch, (10, 90))
            t_low, t_high = np.percentile(t_ch, (10, 90))
            w_std = (w_high - w_low) / 2.563
            t_std = (t_high - t_low) / 2.563

            if w_std > 1.0 and t_std > 1.0:
                channel = warped_lab[:, :, c]
                scale = np.clip(t_std / max(w_std, 0.1), 0.65, 1.55)
                matched = (channel - w_center) * scale + t_center
                detail = channel - cv2.GaussianBlur(channel, (0, 0), 3)
                matched_with_detail = matched + detail * 0.20
                matched_with_detail = np.clip(matched_with_detail, 0, 255)

                alpha = 0.78 if c == 0 else 0.88
                blended = (channel * (1.0 - mask_float * alpha) +
                           matched_with_detail * (mask_float * alpha))
                result_lab[:, :, c] = np.clip(blended, 0, 255)
            else:
                result_lab[:, :, c] = warped_lab[:, :, c]

        # Transfer low-frequency lighting from the target while retaining the
        # source face's high-frequency detail and texture. Center each field on
        # the sampled skin so this complements (rather than repeats) global
        # color matching above.
        if landmarks is not None:
            lighting_landmarks = np.asarray(landmarks, dtype=np.float32)
            lighting_eye_distance = np.linalg.norm(
                np.mean(lighting_landmarks[42:48], axis=0) -
                np.mean(lighting_landmarks[36:42], axis=0)
            )
        else:
            lighting_eye_distance = min(target_img.shape[:2]) * 0.3
        sigma = max(3.0, float(lighting_eye_distance) * 0.18)
        for channel_idx, max_delta, strength in ((0, 35.0, 0.8),
                                                   (1, 10.0, 0.35),
                                                   (2, 10.0, 0.35)):
            target_low = cv2.GaussianBlur(
                target_lab[:, :, channel_idx], (0, 0), sigma
            )
            source_low = cv2.GaussianBlur(
                warped_lab[:, :, channel_idx], (0, 0), sigma
            )
            residual = target_low - source_low
            residual -= np.median(residual[interior])
            residual = np.clip(residual, -max_delta, max_delta)
            result_lab[:, :, channel_idx] += residual * mask_float * strength

        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        result = np.clip(result, 0, 255).astype(np.uint8)

        output = full_warped.copy()
        output[y0:y1, x0:x1] = result
        return output

    def _blend_face(self, target_img, warped_face, mask, landmarks, original_img=None,
                    debug_path=None, face_idx=0, fast=False):
        if fast:
            result = self._multi_band_blend(target_img, warped_face, mask, levels=3)
        else:
            poisson_result = self._poisson_blend(target_img, warped_face, mask, landmarks)

            if poisson_result is not None:
                result = poisson_result
            else:
                if self.debug and debug_path:
                    print(f"  Poisson blend failed, using multi-band blend")
                result = self._multi_band_blend(target_img, warped_face, mask)

        if original_img is not None:
            result = self._restore_eye_voids(result, warped_face, original_img, landmarks)
            result = self._restore_mouth_interior(result, original_img, landmarks)

            if not fast:
                result = self._restore_glasses(result, original_img, landmarks)

        if self.debug and debug_path:
            debug_dir = Path(debug_path).parent
            save_image(result, str(debug_dir / f"debug_{face_idx+1:02d}g_blended.jpg"))

        return result

    @staticmethod
    def _mouth_openness(landmarks):
        gaps = (np.linalg.norm(landmarks[61] - landmarks[67]) +
                np.linalg.norm(landmarks[62] - landmarks[66]) +
                np.linalg.norm(landmarks[63] - landmarks[65])) / 3.0
        width = np.linalg.norm(landmarks[54] - landmarks[48])
        return gaps / max(width, 1.0)

    def _restore_mouth_interior(self, result_img, original_img, landmarks,
                                threshold=0.10):
        openness = self._mouth_openness(landmarks)
        if openness < threshold:
            return result_img

        h, w = result_img.shape[:2]
        mouth_w = np.linalg.norm(landmarks[54] - landmarks[48])

        inner = landmarks[60:68].astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(inner), 255)

        grow = max(3, int(mouth_w * 0.03) | 1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)),
                          iterations=1)
        blur = max(5, int(mouth_w * 0.05) | 1)
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)

        m3 = np.dstack([mask, mask, mask]).astype(np.float32) / 255.0
        out = (result_img.astype(np.float32) * (1.0 - m3) +
               original_img.astype(np.float32) * m3)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _restore_eye_voids(self, result_img, warped_face, original_img, landmarks):
        h, w = result_img.shape[:2]
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(eye_mask, landmarks[36:42].astype(np.int32), 255)
        cv2.fillConvexPoly(eye_mask, landmarks[42:48].astype(np.int32), 255)
        eye_mask = cv2.dilate(eye_mask, np.ones((3, 3), np.uint8), iterations=1)

        warped_gray = cv2.cvtColor(warped_face, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        voids = ((warped_gray < 18) & (target_gray > 40) & (eye_mask > 0)).astype(np.uint8) * 255

        if np.sum(voids > 0) < 8:
            return result_img

        voids = cv2.dilate(voids, np.ones((3, 3), np.uint8), iterations=1)
        voids = cv2.GaussianBlur(voids, (5, 5), 0)

        m3 = np.dstack([voids, voids, voids]).astype(np.float32) / 255.0
        out = (result_img.astype(np.float32) * (1.0 - m3) +
               original_img.astype(np.float32) * m3)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _restore_glasses(self, blended_img, original_img, landmarks):
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        nose_bridge = landmarks[27:31]

        eye_dist = np.linalg.norm(right_eye_center - left_eye_center)

        roi_x = int(min(left_eye_center[0] - eye_dist * 0.5,
                        nose_bridge[:, 0].min()) - eye_dist * 0.1)

        highest_eye_y = min(landmarks[37, 1], landmarks[38, 1], landmarks[43, 1], landmarks[44, 1])
        roi_y = int(highest_eye_y - eye_dist * 0.15)

        roi_x2 = int(max(right_eye_center[0] + eye_dist * 0.5,
                         nose_bridge[:, 0].max()) + eye_dist * 0.1)
        roi_y2 = int(landmarks[30, 1])

        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_x2 = min(original_img.shape[1], roi_x2)
        roi_y2 = min(original_img.shape[0], roi_y2)

        if roi_x2 <= roi_x or roi_y2 <= roi_y:
            return blended_img

        orig_roi = original_img[roi_y:roi_y2, roi_x:roi_x2]
        blend_roi = blended_img[roi_y:roi_y2, roi_x:roi_x2]

        lab_roi = cv2.cvtColor(orig_roi, cv2.COLOR_RGB2LAB)
        l_channel = lab_roi[:, :, 0].astype(np.uint8)

        blurred_l = cv2.GaussianBlur(l_channel, (3, 3), 0)

        _, otsu_mask = cv2.threshold(blurred_l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        total_dark_pixels = np.sum(otsu_mask == 255)
        if total_dark_pixels < 150:
            return blended_img

        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, kernel_line)
        _, thin_lines_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)

        dark_mask = cv2.bitwise_or(otsu_mask, thin_lines_mask)

        kernel_close = np.ones((3, 3), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN,
                                     np.ones((2, 2), np.uint8), iterations=1)

        local_landmarks = landmarks.copy()
        local_landmarks[:, 0] -= roi_x
        local_landmarks[:, 1] -= roi_y

        eye_exclusion = np.zeros_like(dark_mask)
        left_eye_poly = local_landmarks[36:42].astype(np.int32)
        right_eye_poly = local_landmarks[42:48].astype(np.int32)
        cv2.fillConvexPoly(eye_exclusion, left_eye_poly, 255)
        cv2.fillConvexPoly(eye_exclusion, right_eye_poly, 255)

        exclusion_dilate = max(10, int(eye_dist * 0.15))
        eye_exclusion = cv2.dilate(eye_exclusion,
                                   np.ones((exclusion_dilate, exclusion_dilate), np.uint8),
                                   iterations=1)

        nose_pts = local_landmarks[31:36].astype(np.int32)
        cv2.fillConvexPoly(eye_exclusion, nose_pts, 255)
        eye_exclusion = cv2.dilate(eye_exclusion,
                                   np.ones((exclusion_dilate, exclusion_dilate), np.uint8),
                                   iterations=1)

        dark_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(eye_exclusion))

        if np.sum(dark_mask > 0) < 50:
            return blended_img

        roi_h, roi_w = dark_mask.shape[:2]

        ramp_x = np.minimum(np.arange(roi_w), np.arange(roi_w)[::-1]).astype(np.float32)
        ramp_y = np.minimum(np.arange(roi_h), np.arange(roi_h)[::-1]).astype(np.float32)

        fade_window = 6.0
        vignette_x = np.clip(ramp_x / fade_window, 0.0, 1.0)
        vignette_y = np.clip(ramp_y / fade_window, 0.0, 1.0)

        vignette_2d = np.outer(vignette_y, vignette_x)

        dark_mask_float = dark_mask.astype(np.float32) / 255.0
        dark_mask_float = dark_mask_float * vignette_2d

        dark_mask_3d = np.dstack([dark_mask_float, dark_mask_float, dark_mask_float])

        restored_roi = blend_roi.copy()
        frame_pixels = dark_mask_3d > 0.5
        restored_roi[frame_pixels] = orig_roi[frame_pixels]

        result = blended_img.copy()
        result[roi_y:roi_y2, roi_x:roi_x2] = restored_roi

        return result

    def _poisson_blend(self, target_img, warped_face, mask, landmarks):
        h, w = target_img.shape[:2]

        interior = mask > 150
        if np.sum(interior) < 100:
            return None

        clone_mask = interior.astype(np.uint8) * 255

        cv2.rectangle(clone_mask, (0, 0), (w - 1, h - 1), 0, thickness=3)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clone_mask = cv2.morphologyEx(clone_mask, cv2.MORPH_CLOSE, kernel)
        clone_mask = cv2.erode(clone_mask, kernel, iterations=2)

        mx, my, mw, mh = cv2.boundingRect(clone_mask)
        if mw > 10 and mh > 10:
            ramp_x = np.minimum(np.arange(mw), np.arange(mw)[::-1]).astype(np.float32)
            ramp_y = np.minimum(np.arange(mh), np.arange(mh)[::-1]).astype(np.float32)

            fade_window = 12.0
            vignette_x = np.clip(ramp_x / fade_window, 0.0, 1.0)
            vignette_y = np.clip(ramp_y / fade_window, 0.0, 1.0)
            vignette_2d = np.outer(vignette_y, vignette_x)

            roi_mask = clone_mask[my:my+mh, mx:mx+mw].astype(np.float32) / 255.0
            smoothed_roi = (roi_mask * vignette_2d * 255.0).astype(np.uint8)

            _, clone_mask[my:my+mh, mx:mx+mw] = cv2.threshold(smoothed_roi, 127, 255, cv2.THRESH_BINARY)

        eye_mask_cutout = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(eye_mask_cutout, landmarks[36:42].astype(np.int32), 255)
        cv2.fillConvexPoly(eye_mask_cutout, landmarks[42:48].astype(np.int32), 255)
        eye_mask_cutout = cv2.dilate(eye_mask_cutout, np.ones((3, 3), np.uint8), iterations=1)
        clone_mask[eye_mask_cutout > 0] = 0

        if np.sum(clone_mask > 0) < 100:
            return None

        moments = cv2.moments(clone_mask)
        if moments["m00"] > 0:
            center = (int(moments["m10"] / moments["m00"]),
                      int(moments["m01"] / moments["m00"]))
        else:
            x, y, w_box, h_box = cv2.boundingRect(clone_mask)
            center = (int(x + w_box / 2), int(y + h_box / 2))

        try:
            target_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
            warped_bgr = cv2.cvtColor(warped_face, cv2.COLOR_RGB2BGR)

            result_bgr = cv2.seamlessClone(
                warped_bgr, target_bgr, clone_mask, center, cv2.MIX_CLONE
            )

            result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            result = np.clip(result, 0, 255).astype(np.uint8)

            result[eye_mask_cutout > 0] = warped_face[eye_mask_cutout > 0]

            return result

        except Exception as e:
            if self.debug:
                print(f"Poisson internal crash prevented: {e}")
            return None

    def _multi_band_blend(self, target_img, warped_face, mask, levels=None):
        h, w = target_img.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return target_img.copy()

        requested_levels = 4 if levels is None else levels
        padding = 2 ** min(requested_levels, 6) * 2
        x0 = max(0, int(xs.min()) - padding)
        x1 = min(w, int(xs.max()) + padding + 1)
        y0 = max(0, int(ys.min()) - padding)
        y1 = min(h, int(ys.max()) + padding + 1)

        target_roi = target_img[y0:y1, x0:x1]
        warped_roi = warped_face[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        roi_h, roi_w = mask_roi.shape
        max_levels = max(1, int(np.log2(min(roi_h, roi_w))) - 3)
        levels = min(6, max(1, requested_levels), max_levels)

        mask_float = mask_roi.astype(np.float32) / 255.0
        target_f = target_roi.astype(np.float32)
        warped_f = warped_roi.astype(np.float32)

        mask_pyr = [mask_float]
        for _ in range(levels):
            mask_pyr.append(cv2.pyrDown(mask_pyr[-1]))

        target_pyr = []
        warped_pyr = []

        t_cur = target_f.copy()
        w_cur = warped_f.copy()

        for _ in range(levels):
            t_next = cv2.pyrDown(t_cur)
            w_next = cv2.pyrDown(w_cur)

            t_up = cv2.pyrUp(t_next, dstsize=(t_cur.shape[1], t_cur.shape[0]))
            w_up = cv2.pyrUp(w_next, dstsize=(w_cur.shape[1], w_cur.shape[0]))

            target_pyr.append(t_cur - t_up)
            warped_pyr.append(w_cur - w_up)

            t_cur = t_next
            w_cur = w_next

        target_pyr.append(t_cur)
        warped_pyr.append(w_cur)

        blended_pyr = []
        for level in range(levels + 1):
            m = cv2.resize(mask_pyr[level],
                          (target_pyr[level].shape[1], target_pyr[level].shape[0]),
                          interpolation=cv2.INTER_LINEAR)
            m_3d = np.dstack([m, m, m])
            blended = warped_pyr[level] * m_3d + target_pyr[level] * (1.0 - m_3d)
            blended_pyr.append(blended)

        result = blended_pyr[-1]
        for level in range(levels - 1, -1, -1):
            result = cv2.pyrUp(result, dstsize=(blended_pyr[level].shape[1], blended_pyr[level].shape[0]))
            result = result + blended_pyr[level]

        output = target_img.copy()
        output[y0:y1, x0:x1] = np.clip(result, 0, 255).astype(np.uint8)
        return output


class FaceSequence:
    HOLD_FRAMES = 4

    def __init__(self, swapper):
        self.swapper = swapper
        self.tracks = []
        self._identity = None

    def _identity_swap(self):
        cached = getattr(self.swapper, "_identity_model", None)
        if cached is None:
            from src.core.identity_swap import IdentitySwap
            cached = IdentitySwap.from_faces(
                self.swapper.maduro_faces,
                self.swapper.maduro_landmarks_list,
            )
            self.swapper._identity_model = cached
        self._identity = cached
        return cached

    def apply(self, image):
        image = np.asarray(image)
        if image.ndim != 3:
            return image
        result = np.array(image, copy=True)
        detections = self._detections(image)
        used = set()
        for detection in detections:
            center = (detection["left"] + detection["right"]) / 2.0
            iod = detection["iod"]
            best_i, best_dist = None, 1e9
            for index, track in enumerate(self.tracks):
                if index in used:
                    continue
                track_center = (track["left"] + track["right"]) / 2.0
                dist = float(np.linalg.norm(center - track_center))
                gate = max(iod, track["iod"]) * 1.6
                if dist < gate and dist < best_dist:
                    best_i, best_dist = index, dist
            if best_i is None:
                self.tracks.append(self._new_track(image, detection))
                used.add(len(self.tracks) - 1)
            else:
                track = self.tracks[best_i]
                jump = best_dist > max(iod, track["iod"]) * 0.45
                alpha = 1.0 if jump else 0.65
                track["left"] = alpha * detection["left"] + (1.0 - alpha) * track["left"]
                track["right"] = alpha * detection["right"] + (1.0 - alpha) * track["right"]
                track["mouth"] = alpha * detection["mouth"] + (1.0 - alpha) * track["mouth"]
                track["kps"] = alpha * detection["kps"] + (1.0 - alpha) * track["kps"]
                track["box"] = alpha * detection["box"] + (1.0 - alpha) * track["box"]
                track["iod"] = iod
                track["profile"] = bool(detection["profile"])
                track["misses"] = 0
                track["hits"] += 1
                used.add(best_i)

        for index, track in enumerate(self.tracks):
            if index not in used:
                track["misses"] += 1
        self.tracks = [track for track in self.tracks if track["misses"] <= self.HOLD_FRAMES]
        # One face should not get a second copy parked on the chest or hand.
        kept = []
        for track in sorted(self.tracks, key=lambda item: -item["iod"]):
            center = (track["left"] + track["right"]) / 2.0
            if any(float(np.linalg.norm(center - (other["left"] + other["right"]) / 2.0)) < other["iod"] * 0.85
                   for other in kept):
                continue
            kept.append(track)
        self.tracks = kept

        identity = self._identity_swap()
        for track in self.tracks:
            if track["hits"] >= 1 and track.get("kps") is not None:
                result = identity.apply(result, track["kps"], track.get("box"))
        return result

    def _detections(self, image):
        found = self.swapper.detector._yunet_rows(image)
        if found is None:
            return []
        height, width = image.shape[:2]
        detections = []
        for row in found:
            if len(row) < 14:
                continue
            points = row[4:14].reshape(5, 2).astype(np.float64)
            order = np.argsort(points[:2, 0])
            left = points[order[0]]
            right = points[order[1]]
            iod = float(np.linalg.norm(right - left))
            if iod < 3.0:
                continue
            if left[0] < 2 or right[0] > width - 3 or min(left[1], right[1]) < 2:
                continue
            if max(left[1], right[1]) > height - 3:
                continue
            mouth = points[3:5].mean(axis=0)
            box = row[:4].astype(np.float64)
            span = float(right[0] - left[0])
            nose_t = float((points[2, 0] - left[0]) / span) if span > 1 else 0.5
            detections.append({
                "left": left,
                "right": right,
                "mouth": mouth,
                "kps": points.copy(),
                "box": box.copy(),
                "iod": max(iod, 8.0),
                "profile": nose_t < 0.22 or nose_t > 0.78,
            })
        return detections

    def _new_track(self, image, detection):
        source_idx = 0
        try:
            iod = detection["iod"]
            left = detection["left"]
            right = detection["right"]
            mouth = detection["mouth"]
            rect_left = int(max(0, min(left[0], right[0]) - iod * 0.6))
            rect_right = int(min(image.shape[1] - 1, max(left[0], right[0]) + iod * 0.6))
            rect_top = int(max(0, min(left[1], right[1]) - iod))
            rect_bottom = int(min(image.shape[0] - 1, mouth[1] + iod))
            if rect_right > rect_left + 4 and rect_bottom > rect_top + 4:
                import dlib
                rect = dlib.rectangle(rect_left, rect_top, rect_right, rect_bottom)
                landmarks = self.swapper.detector.get_landmarks(image, rect)
                source_idx = self.swapper._select_best_source(landmarks)
        except Exception:
            source_idx = 0
        return {
            "left": detection["left"].copy(),
            "right": detection["right"].copy(),
            "mouth": detection["mouth"].copy(),
            "iod": detection["iod"],
            "misses": 0,
            "hits": 1,
            "source": int(source_idx),
            "profile": bool(detection["profile"]),
            "kps": detection["kps"].copy(),
            "box": detection["box"].copy(),
        }

    def _fit_landmarks(self, image, track):
        import dlib
        iod = float(track["iod"])
        left, right, mouth = track["left"], track["right"], track["mouth"]
        height, width = image.shape[:2]
        rect_left = int(max(0, min(left[0], right[0]) - iod * 0.55))
        rect_right = int(min(width - 1, max(left[0], right[0]) + iod * 0.55))
        rect_top = int(max(0, min(left[1], right[1]) - iod * 0.95))
        rect_bottom = int(min(height - 1, float(mouth[1]) + iod * 1.1))
        if rect_right <= rect_left + 8 or rect_bottom <= rect_top + 8:
            return None
        landmarks = self.swapper.detector.get_landmarks(
            image, dlib.rectangle(rect_left, rect_top, rect_right, rect_bottom),
        ).astype(np.float64)
        if landmarks.shape[0] < 68:
            return None
        eye_center = (landmarks[36:42].mean(axis=0) + landmarks[42:48].mean(axis=0)) / 2.0
        tracked_center = (left + right) / 2.0
        if float(np.linalg.norm(eye_center - tracked_center)) > iod * 0.4:
            return None
        left_eye = landmarks[36:42].mean(axis=0)
        if not (left_eye[1] < landmarks[30, 1] < landmarks[51, 1] < landmarks[8, 1]):
            return None
        height, width = image.shape[:2]
        if landmarks[:, 0].min() < 2 or landmarks[:, 1].min() < 2:
            return None
        if landmarks[:, 0].max() > width - 3 or landmarks[:, 1].max() > height - 3:
            return None
        return landmarks

    def _reshape(self, result, image, track):
        if track.get("profile"):
            return
        fresh = self._fit_landmarks(image, track)
        eye_center = (track["left"] + track["right"]) / 2.0
        if fresh is not None:
            track["landmarks"] = fresh
            track["eye_anchor"] = eye_center.copy()
            landmarks = fresh
        elif track.get("landmarks") is not None and track.get("eye_anchor") is not None:
            shift = eye_center - track["eye_anchor"]
            if float(np.linalg.norm(shift)) > float(track["iod"]) * 0.7:
                return
            landmarks = track["landmarks"] + shift
            track["landmarks"] = landmarks
            track["eye_anchor"] = eye_center.copy()
        else:
            return

        source_lm = self.swapper.maduro_landmarks_list[track["source"]]
        dest = self.swapper._compose_shape(landmarks, source_lm, strength=1.0)
        edited = self.swapper._remap_mesh(
            image, landmarks, dest, canvas=image, lock_ring=True,
        )
        if edited is None:
            return
        warped, coverage = edited
        if int(np.count_nonzero(coverage)) < 40:
            return
        hull = cv2.convexHull(np.round(np.vstack([
            dest[0:17], dest[17:27], self.swapper._forehead_points(dest),
        ])).astype(np.int32))
        face = np.zeros(result.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(face, hull, 255)
        use = (coverage > 0) & (face > 0)
        if int(np.count_nonzero(use)) < 40:
            return
        result[use] = warped[use]
