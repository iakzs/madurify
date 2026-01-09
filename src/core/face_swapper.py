import cv2
import numpy as np
from scipy.spatial import Delaunay
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
            
            warped_face, mask = self._warp_face(
                source_img,
                source_landmarks,
                target_image,
                target_landmarks
            )
            
            if self.debug and debug_path:
                debug_dir = Path(debug_path).parent
                save_image(warped_face, str(debug_dir / "debug_03_warped_before_color.jpg"))
                mask_vis = (mask.astype(np.float32) / 255.0 * 255).astype(np.uint8)
                save_image(cv2.merge([mask_vis, mask_vis, mask_vis]), str(debug_dir / "debug_04_mask.jpg"))
            
            warped_face = self._correct_colors(target_image, warped_face, mask)
            
            if self.debug and debug_path:
                debug_dir = Path(debug_path).parent
                save_image(warped_face, str(debug_dir / "debug_05_warped_after_color.jpg"))
            
            result = self._blend_face(result, warped_face, mask, target_landmarks, debug_path)
            
            if self.debug and debug_path:
                debug_dir = Path(debug_path).parent
                save_image(result, str(debug_dir / "debug_06_final_result.jpg"))
        
        return result
    
    def _select_best_source(self, target_landmarks):
        if len(self.maduro_faces) == 1:
            return 0
        
        target_eye_distance = np.linalg.norm(target_landmarks[36] - target_landmarks[45])
        best_idx = 0
        best_score = float('inf')
        
        for idx, source_landmarks in enumerate(self.maduro_landmarks_list):
            source_eye_distance = np.linalg.norm(source_landmarks[36] - source_landmarks[45])
            score = abs(target_eye_distance - source_eye_distance) / target_eye_distance
            if score < best_score:
                best_score = score
                best_idx = idx
        
        return best_idx
    
    def _warp_face(self, source_img, source_landmarks, target_img, target_landmarks):
        h, w = target_img.shape[:2]
        
        source_points = source_landmarks.astype(np.float32)
        target_points = target_landmarks.astype(np.float32)
        
        hull = cv2.convexHull(target_points.astype(np.int32))
        rect = cv2.boundingRect(hull)
        x, y, w_rect, h_rect = rect
        
        expanded_x = max(0, x - w_rect // 2)
        expanded_y = max(0, y - h_rect // 2)
        expanded_w = min(w, x + w_rect + w_rect // 2) - expanded_x
        expanded_h = min(h, y + h_rect + h_rect // 2) - expanded_y
        
        boundary_points = np.array([
            [expanded_x, expanded_y],
            [expanded_x + expanded_w, expanded_y],
            [expanded_x + expanded_w, expanded_y + expanded_h],
            [expanded_x, expanded_y + expanded_h]
        ], dtype=np.float32)
        
        source_hull = cv2.convexHull(source_points.astype(np.int32))
        source_rect = cv2.boundingRect(source_hull)
        sx, sy, sw_rect, sh_rect = source_rect
        
        source_expanded_x = max(0, sx - sw_rect // 2)
        source_expanded_y = max(0, sy - sh_rect // 2)
        source_expanded_w = min(source_img.shape[1], sx + sw_rect + sw_rect // 2) - source_expanded_x
        source_expanded_h = min(source_img.shape[0], sy + sh_rect + sh_rect // 2) - source_expanded_y
        
        source_boundary = np.array([
            [source_expanded_x, source_expanded_y],
            [source_expanded_x + source_expanded_w, source_expanded_y],
            [source_expanded_x + source_expanded_w, source_expanded_y + source_expanded_h],
            [source_expanded_x, source_expanded_y + source_expanded_h]
        ], dtype=np.float32)
        
        all_source_points = np.vstack([source_points, source_boundary])
        all_target_points = np.vstack([target_points, boundary_points])
        
        tri = Delaunay(all_target_points)
        
        warped = np.zeros((h, w, 3), dtype=np.uint8)
        
        for simplex in tri.simplices:
            src_tri = all_source_points[simplex]
            dst_tri = all_target_points[simplex]
            
            if len(src_tri) < 3 or len(dst_tri) < 3:
                continue
            
            src_rect = cv2.boundingRect(src_tri.astype(np.int32))
            dst_rect = cv2.boundingRect(dst_tri.astype(np.int32))
            
            src_x, src_y, src_w, src_h = src_rect
            dst_x, dst_y, dst_w, dst_h = dst_rect
            
            if src_w < 2 or src_h < 2 or dst_w < 2 or dst_h < 2:
                continue
            
            if src_x < 0 or src_y < 0 or src_x + src_w > source_img.shape[1] or src_y + src_h > source_img.shape[0]:
                continue
            if dst_x < 0 or dst_y < 0 or dst_x + dst_w > w or dst_y + dst_h > h:
                continue
            
            src_tri_shifted = src_tri - [src_x, src_y]
            dst_tri_shifted = dst_tri - [dst_x, dst_y]
            
            src_crop = source_img[src_y:src_y+src_h, src_x:src_x+src_w]
            
            if src_crop.size == 0 or src_crop.shape[0] == 0 or src_crop.shape[1] == 0:
                continue
            
            if len(src_tri_shifted) < 3 or len(dst_tri_shifted) < 3:
                continue
            
            transform = cv2.getAffineTransform(
                src_tri_shifted[:3].astype(np.float32),
                dst_tri_shifted[:3].astype(np.float32)
            )
            
            if transform is None:
                continue
            
            warped_tri = cv2.warpAffine(
                src_crop,
                transform,
                (dst_w, dst_h),
                borderMode=cv2.BORDER_REPLICATE,
                flags=cv2.INTER_CUBIC
            )
            
            if warped_tri.size == 0:
                continue
            
            tri_mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
            cv2.fillConvexPoly(tri_mask, dst_tri_shifted.astype(np.int32), 255)
            
            tri_mask_3d = tri_mask[:, :, np.newaxis] / 255.0
            
            if dst_y + dst_h > h or dst_x + dst_w > w:
                continue
            
            dst_y_end = min(dst_y + dst_h, h)
            dst_x_end = min(dst_x + dst_w, w)
            actual_h = dst_y_end - dst_y
            actual_w = dst_x_end - dst_x
            
            if actual_h <= 0 or actual_w <= 0:
                continue
            
            tri_mask_crop = tri_mask[:actual_h, :actual_w]
            warped_tri_crop = warped_tri[:actual_h, :actual_w]
            tri_mask_3d_crop = tri_mask_crop[:, :, np.newaxis] / 255.0
            
            existing = warped[dst_y:dst_y_end, dst_x:dst_x_end]
            
            if existing.shape != warped_tri_crop.shape:
                continue
            
            valid_pixels = (warped_tri_crop.sum(axis=2) > 20)
            valid_mask_3d = valid_pixels[:, :, np.newaxis].astype(np.float32)
            combined_mask = tri_mask_3d_crop * valid_mask_3d
            
            warped[dst_y:dst_y_end, dst_x:dst_x_end] = (
                existing * (1 - combined_mask) + warped_tri_crop * combined_mask
            ).astype(np.uint8)
        
        face_outline_indices = list(range(17))
        face_outline = target_points[face_outline_indices]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, face_outline.astype(np.int32), 255)
        
        left_eye_points = target_points[36:42].astype(np.int32)
        right_eye_points = target_points[42:48].astype(np.int32)
        nose_points = target_points[27:36].astype(np.int32)
        mouth_points = target_points[48:68].astype(np.int32)
        
        cv2.fillConvexPoly(mask, left_eye_points, 255)
        cv2.fillConvexPoly(mask, right_eye_points, 255)
        cv2.fillConvexPoly(mask, nose_points, 255)
        cv2.fillConvexPoly(mask, mouth_points, 255)
        
        eyebrow_left = target_points[17:22].astype(np.int32)
        eyebrow_right = target_points[22:27].astype(np.int32)
        cv2.fillConvexPoly(mask, eyebrow_left, 255)
        cv2.fillConvexPoly(mask, eyebrow_right, 255)
        
        eye_skin_left = np.array([
            target_points[17],
            target_points[18],
            target_points[36],
            target_points[37],
            target_points[38],
            target_points[39],
            target_points[40],
            target_points[41],
            target_points[31]
        ], dtype=np.int32)
        
        eye_skin_right = np.array([
            target_points[22],
            target_points[23],
            target_points[42],
            target_points[43],
            target_points[44],
            target_points[45],
            target_points[46],
            target_points[47],
            target_points[35]
        ], dtype=np.int32)
        
        cv2.fillConvexPoly(mask, eye_skin_left, 255)
        cv2.fillConvexPoly(mask, eye_skin_right, 255)
        
        forehead_points = np.array([
            target_points[17],
            target_points[18],
            target_points[19],
            target_points[24],
            target_points[25],
            target_points[26],
            target_points[19]
        ], dtype=np.int32)
        cv2.fillConvexPoly(mask, forehead_points, 255)
        
        cheeks_left = np.array([
            target_points[0],
            target_points[1],
            target_points[2],
            target_points[3],
            target_points[4],
            target_points[5],
            target_points[48],
            target_points[31]
        ], dtype=np.int32)
        cheeks_right = np.array([
            target_points[12],
            target_points[13],
            target_points[14],
            target_points[15],
            target_points[16],
            target_points[54],
            target_points[35]
        ], dtype=np.int32)
        cv2.fillConvexPoly(mask, cheeks_left, 255)
        cv2.fillConvexPoly(mask, cheeks_right, 255)
        
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        
        return warped, mask
    
    def _correct_colors(self, target_img, warped_face, mask):
        mask_binary = (mask > 127).astype(np.uint8)
        
        if np.sum(mask_binary) < 50:
            return warped_face
        
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)
        warped_lab = cv2.cvtColor(warped_face, cv2.COLOR_RGB2LAB)
        
        for c in range(3):
            target_channel = target_lab[:, :, c]
            warped_channel = warped_lab[:, :, c]
            
            target_pixels = target_channel[mask_binary > 0]
            warped_pixels = warped_channel[mask_binary > 0]
            
            if len(target_pixels) > 10 and len(warped_pixels) > 10:
                target_mean = np.mean(target_pixels)
                warped_mean = np.mean(warped_pixels)
                target_std = np.std(target_pixels)
                warped_std = np.std(warped_pixels)
                
                if warped_std > 0.1 and target_std > 0.1:
                    if c == 0:
                        gain_mean = np.clip(target_mean / warped_mean, 0.85, 1.15)
                        gain_std = np.clip(target_std / warped_std, 0.85, 1.15)
                    else:
                        gain_mean = np.clip(target_mean / warped_mean, 0.8, 1.2)
                        gain_std = np.clip(target_std / warped_std, 0.8, 1.2)
                    
                    warped_channel = warped_channel.astype(np.float32)
                    warped_channel = (warped_channel - warped_mean) * gain_std + target_mean
                    warped_lab[:, :, c] = np.clip(warped_channel, 0, 255).astype(np.uint8)
        
        warped_corrected = cv2.cvtColor(warped_lab, cv2.COLOR_LAB2RGB)
        
        target_rgb = target_img.astype(np.float32)
        warped_rgb = warped_corrected.astype(np.float32)
        
        target_mean_rgb = np.mean(target_rgb[mask_binary > 0], axis=0)
        warped_mean_rgb = np.mean(warped_rgb[mask_binary > 0], axis=0)
        
        diff = target_mean_rgb - warped_mean_rgb
        diff = np.clip(diff, -30, 30)
        
        mask_3d = mask_binary[:, :, np.newaxis].astype(np.float32) / 255.0
        mask_blur = cv2.GaussianBlur(mask_3d, (21, 21), 0)
        if len(mask_blur.shape) == 2:
            mask_blur_3d = cv2.merge([mask_blur, mask_blur, mask_blur])
        else:
            mask_blur_3d = cv2.merge([mask_blur[:, :, 0], mask_blur[:, :, 0], mask_blur[:, :, 0]])
        
        diff_3d = diff.reshape(1, 1, 3)
        warped_corrected = warped_corrected.astype(np.float32)
        warped_corrected = warped_corrected + diff_3d * mask_blur_3d
        warped_corrected = np.clip(warped_corrected, 0, 255).astype(np.uint8)
        
        return warped_corrected
    
    def _blend_face(self, target_img, warped_face, mask, target_landmarks, debug_path=None):
        mask_binary = (mask > 127).astype(np.uint8)
        
        non_zero_pixels = np.sum(mask_binary > 0)
        if non_zero_pixels < 100:
            if self.debug and debug_path:
                debug_dir = Path(debug_path).parent
                save_image(target_img, str(debug_dir / "debug_fallback_mask_too_small.jpg"))
            return target_img
        
        warped_has_content = (warped_face.sum(axis=2) > 30)
        content_mask = (mask_binary > 0) & warped_has_content
        
        if np.sum(content_mask) < 50:
            if self.debug and debug_path:
                debug_dir = Path(debug_path).parent
                save_image(target_img, str(debug_dir / "debug_fallback_no_content.jpg"))
            return target_img
        
        if self.debug and debug_path:
            debug_dir = Path(debug_path).parent
            warped_face_masked = warped_face.copy()
            warped_face_masked[mask_binary == 0] = target_img[mask_binary == 0]
            save_image(warped_face_masked, str(debug_dir / "debug_warped_clean.jpg"))
        
        mask_refined = cv2.GaussianBlur(mask_binary.astype(np.float32) * 255, (17, 17), 0)
        mask_soft = mask_refined / 255.0
        mask_3d = cv2.merge([mask_soft, mask_soft, mask_soft])
        
        result = (
            target_img.astype(np.float32) * (1.0 - mask_3d) +
            warped_face.astype(np.float32) * mask_3d
        ).astype(np.uint8)
        
        if self.debug and debug_path:
            debug_dir = Path(debug_path).parent
            save_image(result, str(debug_dir / "debug_07_direct_blend.jpg"))
            save_image(result, str(debug_dir / "debug_08_seamless_result.jpg"))
        
        return result
