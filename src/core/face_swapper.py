import cv2
import numpy as np
from pathlib import Path
from .face_detector import FaceDetector
from .image_utils import load_image, save_image


class FaceSwapper:
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
    
    def swap_face(self, target_image, debug_path=None):
        target_image = np.array(target_image) if not isinstance(target_image, np.ndarray) else target_image
        
        faces = self.detector.detect_faces(target_image)
        if len(faces) == 0:
            raise ValueError("No face detected in target image")
        
        result = target_image.copy()
        
        for face_rect in faces:
            target_landmarks = self.detector.get_landmarks(target_image, face_rect)
            
            best_source_idx = self._select_best_source(target_landmarks)
            source_img = self.maduro_faces[best_source_idx]
            source_landmarks = self.maduro_landmarks_list[best_source_idx]
            
            if self.debug and debug_path:
                debug_dir = Path(debug_path).parent
                save_image(source_img, str(debug_dir / "debug_01_source_face.jpg"))
                save_image(target_image, str(debug_dir / "debug_02_target_face.jpg"))

            warped_face, transform_matrix, transformed_landmarks = self._warp_face(
                source_img,
                source_landmarks,
                target_image,
                target_landmarks
            )
            
            if self.debug and debug_path:
                debug_dir = Path(debug_path).parent
                save_image(warped_face, str(debug_dir / "debug_03_warped_before_color.jpg"))

            mask = self._create_face_mask(transformed_landmarks, target_image.shape[:2])
            
            if self.debug and debug_path:
                debug_dir = Path(debug_path).parent
                mask_vis = mask.copy()
                save_image(cv2.merge([mask_vis, mask_vis, mask_vis]), str(debug_dir / "debug_04_mask.jpg"))

            warped_face = self._correct_colors(target_image, warped_face, mask)
            
            if self.debug and debug_path:
                debug_dir = Path(debug_path).parent
                save_image(warped_face, str(debug_dir / "debug_05_warped_after_color.jpg"))

            result = self._blend_face_seamless(result, warped_face, mask, transformed_landmarks, debug_path)
            
            if self.debug and debug_path:
                debug_dir = Path(debug_path).parent
                save_image(result, str(debug_dir / "debug_06_final_result.jpg"))
        
        return result
    
    def _select_best_source(self, target_landmarks):
        if len(self.maduro_faces) == 1:
            return 0

        target_left_eye = np.mean(target_landmarks[36:42], axis=0)
        target_right_eye = np.mean(target_landmarks[42:48], axis=0)
        target_nose = target_landmarks[30]
        target_chin = target_landmarks[8]

        target_eye_center = (target_left_eye + target_right_eye) / 2
        target_face_width = np.linalg.norm(target_right_eye - target_left_eye)
        target_nose_offset = (target_nose[0] - target_eye_center[0]) / target_face_width
        
        best_idx = 0
        best_score = float('inf')
        
        for idx, source_landmarks in enumerate(self.maduro_landmarks_list):
            src_left_eye = np.mean(source_landmarks[36:42], axis=0)
            src_right_eye = np.mean(source_landmarks[42:48], axis=0)
            src_nose = source_landmarks[30]
            
            src_eye_center = (src_left_eye + src_right_eye) / 2
            src_face_width = np.linalg.norm(src_right_eye - src_left_eye)
            src_nose_offset = (src_nose[0] - src_eye_center[0]) / src_face_width

            pose_diff = abs(target_nose_offset - src_nose_offset)
            score = pose_diff
            
            if score < best_score:
                best_score = score
                best_idx = idx
        
        return best_idx
    
    def _compute_similarity_transform(self, src_points, dst_points):
        src_centroid = np.mean(src_points, axis=0)
        dst_centroid = np.mean(dst_points, axis=0)

        src_centered = src_points - src_centroid
        dst_centered = dst_points - dst_centroid

        src_norm = np.sqrt(np.sum(src_centered ** 2))
        dst_norm = np.sqrt(np.sum(dst_centered ** 2))
        scale = dst_norm / src_norm

        src_normalized = src_centered / src_norm
        dst_normalized = dst_centered / dst_norm

        H = src_normalized.T @ dst_normalized
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # dst = scale * R @ src + translation
        M = np.zeros((2, 3), dtype=np.float64)
        M[:2, :2] = scale * R
        M[:, 2] = dst_centroid - scale * R @ src_centroid
        
        return M
    
    def _warp_face(self, source_img, source_landmarks, target_img, target_landmarks):
        h, w = target_img.shape[:2]
        
        source_points = source_landmarks.astype(np.float64)
        target_points = target_landmarks.astype(np.float64)
        
        src_left_eye = np.mean(source_points[36:42], axis=0)
        src_right_eye = np.mean(source_points[42:48], axis=0)
        src_nose = source_points[30]
        src_chin = source_points[8]
        src_brow_left = source_points[17]
        src_brow_right = source_points[26]
        src_jaw_left = source_points[0]
        src_jaw_right = source_points[16]
        
        tgt_left_eye = np.mean(target_points[36:42], axis=0)
        tgt_right_eye = np.mean(target_points[42:48], axis=0)
        tgt_nose = target_points[30]
        tgt_chin = target_points[8]
        tgt_brow_left = target_points[17]
        tgt_brow_right = target_points[26]
        tgt_jaw_left = target_points[0]
        tgt_jaw_right = target_points[16]
        
        src_eye_center = (src_left_eye + src_right_eye) / 2
        tgt_eye_center = (tgt_left_eye + tgt_right_eye) / 2
        
        src_brow_center = (src_brow_left + src_brow_right) / 2
        tgt_brow_center = (tgt_brow_left + tgt_brow_right) / 2
        
        src_face_height = np.linalg.norm(src_chin - src_brow_center)
        tgt_face_height = np.linalg.norm(tgt_chin - tgt_brow_center)
        
        src_face_width = np.linalg.norm(src_jaw_right - src_jaw_left)
        tgt_face_width = np.linalg.norm(tgt_jaw_right - tgt_jaw_left)
        
        scale_h = tgt_face_height / src_face_height
        scale_w = tgt_face_width / src_face_width
        scale = (scale_h + scale_w) / 2
        
        src_angle = np.arctan2(src_right_eye[1] - src_left_eye[1], 
                               src_right_eye[0] - src_left_eye[0])
        tgt_angle = np.arctan2(tgt_right_eye[1] - tgt_left_eye[1], 
                               tgt_right_eye[0] - tgt_left_eye[0])
        rotation = tgt_angle - src_angle
        
        src_center = (src_eye_center + src_nose + src_chin) / 3
        tgt_center = (tgt_eye_center + tgt_nose + tgt_chin) / 3
        
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        
        M = np.zeros((2, 3), dtype=np.float64)
        M[0, 0] = scale * cos_r
        M[0, 1] = -scale * sin_r
        M[1, 0] = scale * sin_r
        M[1, 1] = scale * cos_r
        
        M[0, 2] = tgt_center[0] - (M[0, 0] * src_center[0] + M[0, 1] * src_center[1])
        M[1, 2] = tgt_center[1] - (M[1, 0] * src_center[0] + M[1, 1] * src_center[1])

        warped = cv2.warpAffine(
            source_img,
            M.astype(np.float32),
            (w, h),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_LANCZOS4
        )

        ones = np.ones((source_points.shape[0], 1), dtype=np.float64)
        source_homogeneous = np.hstack([source_points, ones])
        rigid_landmarks = (source_homogeneous @ M.T).astype(np.float32)

        warped = self._apply_expression_warp(
            warped,
            rigid_landmarks,
            target_points.astype(np.float32),
            (h, w),
        )

        return warped, M, target_points.astype(np.float32)

    def _apply_expression_warp(self, img, src_landmarks, tgt_landmarks, img_size):
        h, w = img_size

        EXPRESSION_IDX = set(range(17, 27)) | set(range(36, 68))

        dest_points = src_landmarks.copy()
        for i in EXPRESSION_IDX:
            dest_points[i] = tgt_landmarks[i]

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

        face_anchors = _bbox_anchors(src_landmarks)   # same for src and dest
        all_src  = np.vstack([src_landmarks,  face_anchors])
        all_dest = np.vstack([dest_points,    face_anchors])

        rect   = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)

        clamped_dest = []
        for pt in all_dest:
            cx = float(np.clip(pt[0], 0, w - 1))
            cy = float(np.clip(pt[1], 0, h - 1))
            clamped_dest.append((cx, cy))
            subdiv.insert((cx, cy))

        dest_map = {(round(cx), round(cy)): i
                    for i, (cx, cy) in enumerate(clamped_dest)}

        output = img.copy()

        for tri in subdiv.getTriangleList():
            pts_dest = np.array(
                [(tri[0], tri[1]), (tri[2], tri[3]), (tri[4], tri[5])],
                dtype=np.float32,
            )

            if any(p[0] < 0 or p[0] >= w or p[1] < 0 or p[1] >= h
                   for p in pts_dest):
                continue

            indices = []
            valid = True
            for pt in pts_dest:
                idx = dest_map.get((round(float(pt[0])), round(float(pt[1]))))
                if idx is None:
                    valid = False
                    break
                indices.append(idx)
            if not valid:
                continue

            pts_src = np.array([all_src[i] for i in indices], dtype=np.float32)
            pts_src[:, 0] = np.clip(pts_src[:, 0], 0, w - 1)
            pts_src[:, 1] = np.clip(pts_src[:, 1], 0, h - 1)

            M_tri = cv2.getAffineTransform(pts_src, pts_dest)

            warped_patch = cv2.warpAffine(
                img, M_tri, (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            tri_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(tri_mask, pts_dest.astype(np.int32), 255)
            output = np.where(tri_mask[:, :, np.newaxis] > 0, warped_patch, output)

        return output

    def _create_face_mask(self, landmarks, img_shape):
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
        forehead_height = face_height * 0.25
        
        inner_jaw = jaw[1:16]
        face_center_x = (jaw[0][0] + jaw[16][0]) / 2
        inward_factor = 0.15
        adjusted_jaw = []
        for pt in inner_jaw:
            new_pt = pt.copy()
            new_pt[0] = pt[0] + (face_center_x - pt[0]) * inward_factor
            adjusted_jaw.append(new_pt)
        adjusted_jaw = np.array(adjusted_jaw)
        
        forehead_pts = []
        all_brow = np.vstack([left_brow, right_brow])
        sorted_idx = np.argsort(all_brow[:, 0])
        sorted_brow = all_brow[sorted_idx]
        for pt in sorted_brow:
            fh_pt = pt + np.array([up_dx * forehead_height, up_dy * forehead_height])
            fh_pt[0] = fh_pt[0] + (face_center_x - fh_pt[0]) * inward_factor
            forehead_pts.append(fh_pt)
        forehead_pts = np.array(forehead_pts)
        
        face_contour = np.vstack([adjusted_jaw, forehead_pts[::-1]])
        hull = cv2.convexHull(face_contour.astype(np.int32))
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        
        blur_size = max(int(face_height * 0.12), 21)
        if blur_size % 2 == 0:
            blur_size += 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        
        return mask
    
    def _correct_colors(self, target_img, warped_face, mask):
        mask_binary = (mask > 127).astype(np.uint8)
        
        if np.sum(mask_binary) < 100:
            return warped_face
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB).astype(np.float64)
        warped_lab = cv2.cvtColor(warped_face, cv2.COLOR_RGB2LAB).astype(np.float64)
        
        mask_region = mask_binary > 0

        target_pixels = target_lab[mask_region]
        warped_pixels = warped_lab[mask_region]
        
        if len(target_pixels) < 100 or len(warped_pixels) < 100:
            return warped_face

        target_mean = np.mean(target_pixels, axis=0)
        target_std = np.std(target_pixels, axis=0)
        warped_mean = np.mean(warped_pixels, axis=0)
        warped_std = np.std(warped_pixels, axis=0)

        result_lab = warped_lab.copy()
        
        for c in range(3):
            if warped_std[c] > 1e-6:
                if c == 0:
                    alpha = 0.35
                else:
                    alpha = 0.25

                transfer_mean = target_mean[c] * alpha + warped_mean[c] * (1 - alpha)
                transfer_std = target_std[c] * alpha + warped_std[c] * (1 - alpha)

                channel = warped_lab[:, :, c]
                normalized = (channel - warped_mean[c]) / (warped_std[c] + 1e-6)
                transferred = np.clip(normalized * transfer_std + transfer_mean, 0, 255)

                result_lab[:, :, c] = np.where(mask_region, transferred, warped_lab[:, :, c])
        
        result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return result
    
    def _blend_face_seamless(self, target_img, warped_face, mask, landmarks, debug_path=None):
        h, w = target_img.shape[:2]
        mask_binary = (mask > 127).astype(np.uint8) * 255
        
        if np.sum(mask_binary > 0) < 100:
            return target_img
        
        nose_tip = landmarks[30]
        chin = landmarks[8]
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        eye_center = (left_eye + right_eye) / 2
        mouth_center = (landmarks[48] + landmarks[54]) / 2
        
        face_center_x = int((left_eye[0] + right_eye[0] + nose_tip[0] + mouth_center[0]) / 4)
        face_center_y = int((eye_center[1] + nose_tip[1] + mouth_center[1]) / 3)
        
        face_center_x = max(1, min(w - 2, face_center_x))
        face_center_y = max(1, min(h - 2, face_center_y))
        center = (face_center_x, face_center_y)
        
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._alpha_blend(target_img, warped_face, mask)
        
        clean_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(clean_mask, contours, -1, 255, -1)
        
        try:
            target_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
            warped_bgr = cv2.cvtColor(warped_face, cv2.COLOR_RGB2BGR)

            result_bgr = cv2.seamlessClone(
                warped_bgr,
                target_bgr,
                clean_mask,
                center,
                cv2.NORMAL_CLONE
            )
            
            result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            
            if self.debug and debug_path:
                debug_dir = Path(debug_path).parent
                save_image(result, str(debug_dir / "debug_07_seamless_clone.jpg"))
            
            return result
            
        except Exception as e:
            if self.debug:
                print(f"Seamless clone failed: {e}, falling back to alpha blend")
            return self._alpha_blend(target_img, warped_face, mask)
    
    def _alpha_blend(self, target_img, warped_face, mask):
        mask_float = mask.astype(np.float32) / 255.0

        mask_soft = cv2.GaussianBlur(mask_float, (21, 21), 0)
        mask_3d = np.dstack([mask_soft, mask_soft, mask_soft])

        result = (
            target_img.astype(np.float32) * (1.0 - mask_3d) +
            warped_face.astype(np.float32) * mask_3d
        ).astype(np.uint8)
        
        return result
