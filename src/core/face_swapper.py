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
            
            warped_face = self._warp_face(
                source_img,
                source_landmarks,
                target_image,
                target_landmarks
            )
            
            if self.debug and debug_path:
                debug_dir = Path(debug_path).parent
                save_image(warped_face, str(debug_dir / "debug_03_warped_before_color.jpg"))
            
            mask = self._create_mask(target_landmarks, target_image.shape[:2])
            
            if self.debug and debug_path:
                debug_dir = Path(debug_path).parent
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
            
            warped[dst_y:dst_y_end, dst_x:dst_x_end] = (
                existing * (1 - tri_mask_3d_crop) + warped_tri_crop * tri_mask_3d_crop
            ).astype(np.uint8)
        
        face_hull = cv2.convexHull(target_points.astype(np.int32))
        face_mask_check = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask_check, face_hull, 255)
        
        face_region_content = warped[face_mask_check > 0]
        if len(face_region_content) > 0 and np.sum(face_region_content) < 10000:
            try:
                left_eye = source_points[36]
                right_eye = source_points[45]
                nose_tip = source_points[30]
                
                left_eye_tgt = target_points[36]
                right_eye_tgt = target_points[45]
                nose_tip_tgt = target_points[30]
                
                src_tri = np.array([left_eye, right_eye, nose_tip], dtype=np.float32)
                dst_tri = np.array([left_eye_tgt, right_eye_tgt, nose_tip_tgt], dtype=np.float32)
                
                M = cv2.getAffineTransform(src_tri, dst_tri)
                warped_affine = cv2.warpAffine(source_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LANCZOS4)
                warped[face_mask_check > 0] = warped_affine[face_mask_check > 0]
            except:
                pass
        
        return warped
    
    def _create_mask(self, target_landmarks, img_shape):
        h, w = img_shape
        target_points = target_landmarks.astype(np.float32)
        
        face_indices = list(range(17))
        face_indices.extend(range(17, 68))
        
        face_region_points = target_points[face_indices]
        full_face_hull = cv2.convexHull(face_region_points.astype(np.int32))
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, full_face_hull.astype(np.int32), 255)
        
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def _correct_colors(self, target_img, warped_face, mask):
        mask_binary = (mask > 127).astype(np.uint8)
        
        if np.sum(mask_binary) < 50:
            return warped_face
        
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        warped_lab = cv2.cvtColor(warped_face, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        mask_region = mask_binary > 0
        
        if np.sum(mask_region) > 100:
            target_pixels_lab = target_lab[mask_region]
            warped_pixels_lab = warped_lab[mask_region]
            
            if len(target_pixels_lab) > 10 and len(warped_pixels_lab) > 10:
                target_mean = np.mean(target_pixels_lab, axis=0)
                warped_mean = np.mean(warped_pixels_lab, axis=0)
                target_std = np.std(target_pixels_lab, axis=0)
                warped_std = np.std(warped_pixels_lab, axis=0)
                
                for c in range(3):
                    target_channel = target_lab[:, :, c]
                    warped_channel = warped_lab[:, :, c]
                    
                    target_mean_c = target_mean[c]
                    warped_mean_c = warped_mean[c]
                    target_std_c = target_std[c]
                    warped_std_c = warped_std[c]
                    
                    if warped_std_c > 0.1 and target_std_c > 0.1:
                        if c == 0:
                            gain_std = np.clip(target_std_c / (warped_std_c + 1e-6), 0.8, 1.2)
                        else:
                            gain_std = np.clip(target_std_c / (warped_std_c + 1e-6), 0.85, 1.15)
                        
                        warped_channel = (warped_channel - warped_mean_c) * gain_std + target_mean_c
                        warped_lab[:, :, c] = np.clip(warped_channel, 0, 255)
        
        warped_corrected = cv2.cvtColor(warped_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        target_rgb = target_img.astype(np.float32)
        warped_rgb = warped_corrected.astype(np.float32)
        
        mask_float = mask_binary.astype(np.float32) / 255.0
        mask_smooth = cv2.GaussianBlur(mask_float, (31, 31), 0)
        mask_3d = cv2.merge([mask_smooth, mask_smooth, mask_smooth])
        
        if np.sum(mask_binary) > 100:
            target_mean_rgb = np.mean(target_rgb[mask_region], axis=0)
            warped_mean_rgb = np.mean(warped_rgb[mask_region], axis=0)
            
            diff = target_mean_rgb - warped_mean_rgb
            diff = np.clip(diff, -20, 20)
            
            diff_3d = diff.reshape(1, 1, 3)
            warped_corrected = warped_corrected.astype(np.float32)
            warped_corrected = warped_corrected + diff_3d * mask_3d
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
        
        if self.debug and debug_path:
            debug_dir = Path(debug_path).parent
            warped_face_masked = warped_face.copy()
            warped_face_masked[mask_binary == 0] = target_img[mask_binary == 0]
            save_image(warped_face_masked, str(debug_dir / "debug_warped_clean.jpg"))
        
        mask_soft = cv2.GaussianBlur(mask_binary.astype(np.float32) * 255, (45, 45), 0) / 255.0
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
