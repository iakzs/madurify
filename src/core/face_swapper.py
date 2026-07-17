import cv2
import numpy as np
from pathlib import Path
from .face_detector import FaceDetector
from .image_utils import load_image, save_image


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
        target_image = np.array(target_image) if not isinstance(target_image, np.ndarray) else target_image

        if faces_override is not None:
            faces = faces_override
        else:
            faces = self.detector.detect_faces(target_image)
            if len(faces) == 0:
                raise ValueError("No face detected in target image")
            faces = sorted(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()), reverse=True)

        result = target_image.copy()
        processed_regions = []

        for face_idx, face_item in enumerate(faces):
            try:
                if faces_override is not None:
                    face_rect, target_landmarks = face_item
                    target_landmarks = np.asarray(target_landmarks, dtype=np.float64)
                else:
                    face_rect = face_item
                    target_landmarks = self.detector.get_landmarks(target_image, face_rect)

                best_source_idx = self._select_best_source(target_landmarks)
                source_img = self.maduro_faces[best_source_idx]
                source_landmarks = self.maduro_landmarks_list[best_source_idx]

                if self.debug and debug_path:
                    debug_dir = Path(debug_path).parent
                    save_image(source_img, str(debug_dir / f"debug_{face_idx+1:02d}a_source.jpg"))
                    save_image(target_image, str(debug_dir / f"debug_{face_idx+1:02d}b_target.jpg"))

                warped_face, M, transformed_landmarks = self._warp_face(
                    source_img, source_landmarks, target_image, target_landmarks
                )
                warped_face = np.clip(warped_face, 0, 255).astype(np.uint8)

                if self.debug and debug_path:
                    debug_dir = Path(debug_path).parent
                    save_image(warped_face, str(debug_dir / f"debug_{face_idx+1:02d}c_warped.jpg"))

                face_center = np.mean(transformed_landmarks, axis=0)

                mask = self._create_face_mask(transformed_landmarks, target_image.shape[:2],
                                              None if fast else target_image)

                if self.debug and debug_path:
                    debug_dir = Path(debug_path).parent
                    mask_vis = mask.copy()
                    save_image(cv2.merge([mask_vis, mask_vis, mask_vis]),
                               str(debug_dir / f"debug_{face_idx+1:02d}d_mask.jpg"))

                warped_face = self._correct_colors(target_image, warped_face, mask)
                warped_face = np.clip(warped_face, 0, 255).astype(np.uint8)

                if self.debug and debug_path:
                    debug_dir = Path(debug_path).parent
                    save_image(warped_face, str(debug_dir / f"debug_{face_idx+1:02d}e_color.jpg"))

                face_radius = max(
                    np.linalg.norm(face_center - transformed_landmarks[0]),
                    np.linalg.norm(face_center - transformed_landmarks[16])
                )
                processed_regions.append((face_center, face_radius))

                result = self._blend_face(result, warped_face, mask, transformed_landmarks,
                                          target_image, debug_path, face_idx, fast=fast)
                result = np.clip(result, 0, 255).astype(np.uint8)

                if self.debug and debug_path:
                    debug_dir = Path(debug_path).parent
                    save_image(result, str(debug_dir / f"debug_{face_idx+1:02d}f_result.jpg"))

            except Exception as e:
                if self.debug:
                    print(f"Warning: face {face_idx} failed: {e}")
                continue

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
                     abs(target_jaw_width - src_jaw_width) * 1.0)

            if score < best_score:
                best_score = score
                best_idx = idx

        return best_idx

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

    def _warp_face(self, source_img, source_landmarks, target_img, target_landmarks):
        from scipy.spatial import Delaunay as ScipyDelaunay
        h, w = target_img.shape[:2]

        source_points = source_landmarks.astype(np.float32)
        target_points = target_landmarks.astype(np.float32)

        M = self._compute_eye_similarity(source_points, target_points)

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
        result = target_img.copy()

        for simplex in tri.simplices:
            src_tri = all_src[simplex]
            dst_tri = all_dst[simplex]

            src_area = ((src_tri[1][0] - src_tri[0][0]) * (src_tri[2][1] - src_tri[0][1]) -
                        (src_tri[1][1] - src_tri[0][1]) * (src_tri[2][0] - src_tri[0][0]))
            dst_area = ((dst_tri[1][0] - dst_tri[0][0]) * (dst_tri[2][1] - dst_tri[0][1]) -
                        (dst_tri[1][1] - dst_tri[0][1]) * (dst_tri[2][0] - dst_tri[0][0]))
            if abs(src_area) < 1.0 or abs(dst_area) < 1.0:
                continue

            x, y, tw, th = cv2.boundingRect(dst_tri)
            x = max(0, x); y = max(0, y)
            tw = min(w - x, tw); th = min(h - y, th)
            if tw < 2 or th < 2:
                continue

            dst_offset = dst_tri - np.float32([[x, y], [x, y], [x, y]])
            mask = np.zeros((th, tw), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.round(dst_offset).astype(np.int32), 255)

            tri_M = cv2.getAffineTransform(src_tri, dst_offset)
            patch = cv2.warpAffine(source_img, tri_M, (tw, th),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

            m3 = np.dstack([mask, mask, mask]).astype(np.float32) / 255.0
            roi = result[y:y+th, x:x+tw].astype(np.float32)
            blended = roi * (1.0 - m3) + patch.astype(np.float32) * m3
            result[y:y+th, x:x+tw] = np.clip(blended, 0, 255).astype(np.uint8)

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
        for i in range(9, 16, -1):
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

    def _correct_colors(self, target_img, warped_face, mask):
        mask_float = mask.astype(np.float32) / 255.0

        base_interior = mask_float > 0.9
        if np.sum(base_interior) < 100:
            return warped_face

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
            return warped_face

        result_lab = warped_lab.copy()

        for c in range(3):
            w_ch = warped_skin[:, c]
            t_ch = target_skin[:, c]

            w_mean, w_std = w_ch.mean(), w_ch.std()
            t_mean, t_std = t_ch.mean(), t_ch.std()

            if w_std > 1.0 and t_std > 1.0:
                channel = warped_lab[:, :, c]
                scale = np.clip(t_std / max(w_std, 0.1), 0.3, 3.0)
                matched = (channel - w_mean) * scale + t_mean
                detail = channel - cv2.GaussianBlur(channel, (0, 0), 3)
                matched_with_detail = matched + detail * 0.20
                matched_with_detail = np.clip(matched_with_detail, 0, 255)

                alpha = 0.92 if c == 0 else 0.82
                blended = (channel * (1.0 - mask_float * alpha) +
                           matched_with_detail * (mask_float * alpha))
                result_lab[:, :, c] = np.clip(blended, 0, 255)
            else:
                result_lab[:, :, c] = warped_lab[:, :, c]

        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

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
        max_levels = int(np.log2(min(h, w))) - 3
        levels = levels if levels is not None else min(4, max_levels)
        levels = max(2, levels)
        levels = min(6, levels)

        mask_float = mask.astype(np.float32) / 255.0
        target_f = target_img.astype(np.float32)
        warped_f = warped_face.astype(np.float32)

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

        return np.clip(result, 0, 255).astype(np.uint8)
