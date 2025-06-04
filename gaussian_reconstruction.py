import torch
import numpy as np
from torch import nn
import cv2
from typing import List, Tuple, Dict
import logging
import sys

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,  # DEBUGからINFOに変更
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gaussian_reconstruction.log')
    ]
)
logger = logging.getLogger(__name__)

# 特定のモジュールのログレベルを設定
logging.getLogger('gaussian_reconstruction').setLevel(logging.INFO)

class GaussianSplatting(nn.Module):
    def __init__(self, num_points: int = 100000):
        super().__init__()
        self.num_points = num_points
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing GaussianSplatting with {num_points} points on {self.device}")
        
        # ガウス点のパラメータ
        # 初期化範囲を狭め、z方向を0.5付近に集中させる
        self.positions = nn.Parameter(torch.randn(num_points, 3, device=self.device) * 0.05 + torch.tensor([0, 0, 0.5], device=self.device))
        self.scales = nn.Parameter(torch.ones(num_points, 3, device=self.device) * 0.01)  # スケールを小さく
        self.rotations = nn.Parameter(
            torch.cat([
                torch.ones(num_points, 1, device=self.device),
                torch.zeros(num_points, 3, device=self.device)
            ], dim=1)
        )
        self.colors = nn.Parameter(torch.rand(num_points, 3, device=self.device))  # ランダムな色
        self.opacities = nn.Parameter(torch.ones(num_points, device=self.device) * 0.8)  # 不透明度を調整
        
        # 最適化パラメータ
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        logger.debug(f"Initial positions shape: {self.positions.shape}")
        logger.debug(f"Initial colors shape: {self.colors.shape}")
        
    def forward(self, camera_poses: torch.Tensor) -> torch.Tensor:
        """
        カメラポーズから画像を生成
        
        Args:
            camera_poses: カメラの位置と姿勢 [batch_size, 4, 4]
            
        Returns:
            生成された画像 [batch_size, height, width, 3]
        """
        logger.debug(f"Forward pass with camera_poses shape: {camera_poses.shape}")
        
        try:
            with torch.no_grad():
                # 3D点を2Dに投影
                projected_points = self.project_points(camera_poses)
                logger.debug(f"Projected points shape: {projected_points.shape}")
                
                # ガウス分布の描画
                rendered_images = self.render_gaussians(projected_points)
                logger.debug(f"Rendered images shape: {rendered_images.shape}")
                
                return rendered_images
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
    
    def project_points(self, camera_poses: torch.Tensor) -> torch.Tensor:
        """
        3D点を2D画像座標に投影（ピンホールカメラモデル）
        """
        try:
            with torch.no_grad():
                # 仮のカメラ内部パラメータ
                fx, fy = 800, 800
                cx, cy = 640, 360  # 画像中心
                # カメラ座標系に変換（ここではワールド座標=カメラ座標と仮定）
                x = self.positions[:, 0].detach()
                y = self.positions[:, 1].detach()
                z = self.positions[:, 2].detach() + 3.0  # カメラ前方にオフセットを大きく
                u = fx * (x / z) + cx
                v = fy * (y / z) + cy
                projected_points = torch.stack([u, v], dim=1)
                # デバッグ用に一部の投影点をログ出力
                logger.info(f"projected_points sample: {projected_points[:10]}")
                return projected_points
        except Exception as e:
            logger.error(f"Error in project_points: {str(e)}")
            raise
    
    def render_gaussians(self, projected_points: torch.Tensor) -> torch.Tensor:
        """ガウス分布の描画"""
        if len(projected_points) == 0:
            logger.warning("No projected points to render")
            return torch.zeros((720, 1280, 3), device=self.device)

        try:
            # 画像の初期化（NumPy配列で）
            image_np = np.zeros((720, 1280, 3), dtype=np.float32)
            valid_points = 0

            with torch.no_grad():
                for i in range(min(self.num_points, len(projected_points))):
                    try:
                        xy = projected_points[i]
                        if len(xy) < 2:
                            continue
                        x, y = xy[0].item(), xy[1].item()
                        if 0 <= x < 1280 and 0 <= y < 720:
                            radius = float(self.scales[i, 0].item() * 100)
                            dx = x - int(x)
                            dy = y - int(y)
                            distance = np.sqrt(dx*dx + dy*dy)
                            weight = np.exp(-distance * distance / (2 * radius * radius))
                            point_color = self.colors[i].detach().cpu().numpy() * weight
                            y_idx = int(y)
                            x_idx = int(x)
                            image_np[y_idx, x_idx] = point_color
                            valid_points += 1
                    except Exception as e:
                        logger.warning(f"Error rendering point {i}: {str(e)}")
                        continue

            if valid_points == 0:
                logger.warning("No valid points rendered inside the image.")
            else:
                logger.info(f"Rendered {valid_points} valid points inside the image.")

            # NumPy配列からtorchテンソルへ変換
            if image_np.ndim == 2:
                image_np = np.stack([image_np]*3, axis=-1)
            elif image_np.shape != (720, 1280, 3):
                image_np = np.reshape(image_np, (720, 1280, 3))
            image = torch.from_numpy(image_np).to(self.device)
            logger.info(f"render_gaussians output shape: {image.shape}")
            return image
        except Exception as e:
            logger.error(f"Error in render_gaussians: {str(e)}")
            return torch.zeros((720, 1280, 3), device=self.device)

class ImageProcessor:
    def __init__(self):
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        logger.info("Initialized ImageProcessor")
        
    def detect_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        画像から特徴点を検出
        
        Args:
            image: 入力画像
            
        Returns:
            keypoints: 特徴点
            descriptors: 特徴量記述子
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            return keypoints, descriptors
        except Exception as e:
            logger.error(f"Error in detect_features: {str(e)}")
            raise
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """
        2つの画像の特徴点をマッチング
        
        Args:
            desc1: 1枚目の特徴量記述子
            desc2: 2枚目の特徴量記述子
            
        Returns:
            matches: マッチング結果
        """
        try:
            if desc1 is None or desc2 is None:
                return []
                
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            return good_matches
        except Exception as e:
            logger.error(f"Error in match_features: {str(e)}")
            return []

class CameraTracker:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.current_pose = None
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.poses = []  # カメラの軌跡を保存
        logger.info("Initialized CameraTracker")
        
    def initialize_camera(self, image_size: Tuple[int, int]):
        """
        カメラパラメータの初期化
        
        Args:
            image_size: 画像サイズ (width, height)
        """
        try:
            # カメラ行列の初期化（仮の値）
            focal_length = max(image_size)
            center = (image_size[0] / 2, image_size[1] / 2)
            self.camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ])
            self.dist_coeffs = np.zeros(5)
            logger.info(f"Initialized camera with size {image_size}")
            logger.debug(f"Camera matrix:\n{self.camera_matrix}")
        except Exception as e:
            logger.error(f"Error in initialize_camera: {str(e)}")
            raise
        
    def track_camera(self, frame: np.ndarray) -> np.ndarray:
        try:
            # 特徴点を検出
            keypoints, descriptors = self.image_processor.detect_features(frame)
            
            if self.prev_frame is None:
                self.prev_frame = frame
                self.prev_keypoints = keypoints
                self.prev_descriptors = descriptors
                self.current_pose = np.eye(4)
                self.poses.append(self.current_pose)
                logger.info("First frame processed")
                return self.current_pose
            
            # 特徴点をマッチング
            matches = self.image_processor.match_features(self.prev_descriptors, descriptors)
            
            if len(matches) < 8:
                logger.warning(f"Not enough matches: {len(matches)}")
                return self.current_pose
            
            # マッチング点の座標を取得
            src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # 基本行列を計算
            E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.camera_matrix)
            
            # カメラの動きを推定
            _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix)
            
            # 相対的な動きを計算
            relative_pose = np.eye(4)
            relative_pose[:3, :3] = R
            relative_pose[:3, 3] = t.ravel()
            
            # 現在の姿勢を更新
            self.current_pose = np.matmul(self.current_pose, relative_pose)
            self.poses.append(self.current_pose)
            
            # 状態を更新
            self.prev_frame = frame
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            
            return self.current_pose
        except Exception as e:
            logger.error(f"Error in track_camera: {str(e)}")
            return self.current_pose

class ObjectTracker:
    def __init__(self):
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.tracked_points = []  # 追跡中の特徴点を保存
        logger.info("Initialized ObjectTracker")
    
    def detect_object(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """対象物の特徴点を検出"""
        try:
            # グレースケール変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 特徴点検出
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            if self.prev_frame is None:
                self.prev_frame = gray
                self.prev_keypoints = keypoints
                self.prev_descriptors = descriptors
                # 最初のフレームの特徴点を保存
                self.tracked_points = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
                return keypoints, descriptors, None
            
            # 特徴点のマッチング
            matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) < 8:
                return keypoints, descriptors, None
            
            # マッチング点の座標を取得
            src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # 基本行列を計算
            E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            # 追跡中の特徴点を現フレームのマッチした点に更新
            self.tracked_points = [(int(keypoints[m.trainIdx].pt[0]), int(keypoints[m.trainIdx].pt[1])) for m in good_matches]
            # 新しい特徴点を追加（最大100点まで）
            if len(self.tracked_points) < 100:
                for kp in keypoints:
                    pt = (int(kp.pt[0]), int(kp.pt[1]))
                    if pt not in self.tracked_points:
                        self.tracked_points.append(pt)
                        if len(self.tracked_points) >= 100:
                            break
            # 状態を更新
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return keypoints, descriptors, E
        except Exception as e:
            logger.error(f"Error in detect_object: {str(e)}")
            return None, None, None

class HighResReconstructor:
    def __init__(self, mosaic_mode=True, blend_mode="min"):
        self.gaussian_model = GaussianSplatting()
        self.camera_tracker = CameraTracker()
        self.object_tracker = ObjectTracker()
        self.frames = []
        self.camera_poses = []
        self.object_poses = []  # 対象物の姿勢を保存
        self.mosaic_mode = mosaic_mode
        self.mosaic_canvas = None
        self.mosaic_transform = np.eye(3, dtype=np.float32)
        self.mosaic_size = (2560, 1440)  # パノラマ用キャンバスサイズ
        self.mosaic_offset = (640, 360)  # 中心オフセット
        self.blend_mode = blend_mode  # 合成方法: median, max, min
        self.label_corners = None  # ラベルの4点
        logger.info("Initialized HighResReconstructor (mosaic_mode={}, blend_mode={})".format(mosaic_mode, blend_mode))
        
    def warp_to_label(self, frame, label_corners, output_size=(400, 400)):
        """
        ラベル領域を正面化するワープ変換
        
        Args:
            frame: 入力画像
            label_corners: ラベルの4点の座標 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            output_size: 出力画像のサイズ (width, height)
            
        Returns:
            warped: ワープ変換後の画像
        """
        try:
            if frame is None or label_corners is None or len(label_corners) != 4:
                logger.warning("Invalid input for warp_to_label")
                return frame
                
            # 入力点をfloat32型に変換
            pts_src = np.array(label_corners, dtype=np.float32)
            pts_dst = np.array([
                [0, 0],
                [output_size[0]-1, 0],
                [output_size[0]-1, output_size[1]-1],
                [0, output_size[1]-1]
            ], dtype=np.float32)
            
            # 変換行列を計算
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            if M is None:
                logger.warning("Failed to compute perspective transform")
                return frame
                
            # 変換行列がfloat32型であることを確認
            M = M.astype(np.float32)
            
            # ワープ変換を実行
            warped = cv2.warpPerspective(frame, M, output_size)
            if warped is None:
                logger.warning("Failed to perform warp perspective")
                return frame
                
            return warped
            
        except Exception as e:
            logger.error(f"Error in warp_to_label: {str(e)}")
            return frame

    def auto_detect_label_corners(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        YOLOv8-segのセグメンテーションマスクからラベルの4点を自動検出
        
        Args:
            frame: 入力画像
            
        Returns:
            corners: ラベルの4点の座標 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n-seg.pt')
            results = model(frame)
            
            if not (hasattr(results[0], 'masks') and results[0].masks is not None):
                logger.warning("No mask detected in frame")
                return None
                
            mask_data = results[0].masks.data.cpu().numpy()
            if mask_data.shape[0] == 0:
                logger.warning("Empty mask in frame")
                return None
                
            # 最大面積のマスクを選択
            areas = mask_data.sum(axis=(1,2))
            idx = areas.argmax()
            mask = (mask_data[idx] * 255).astype(np.uint8)
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            
            # 輪郭を検出
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logger.warning("No contours found in mask")
                return None
                
            # 最大の輪郭を選択
            max_contour = max(contours, key=cv2.contourArea)
            
            # 輪郭を近似して4点を取得
            epsilon = 0.02 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
            
            if len(approx) != 4:
                # 4点でない場合は最小外接矩形を使用
                rect = cv2.minAreaRect(max_contour)
                box = cv2.boxPoints(rect)
                approx = np.int0(box)
            
            # 左上→右上→右下→左下の順に並び替え
            corners = approx.reshape(4, 2)
            center = np.mean(corners, axis=0)
            angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
            sorted_idx = np.argsort(angles)
            corners = corners[sorted_idx]
            
            logger.info(f"Auto-detected label corners: {corners.tolist()}")
            return corners.tolist()
            
        except Exception as e:
            logger.error(f"Error in auto_detect_label_corners: {str(e)}")
            return None

    def add_frame(self, frame: np.ndarray):
        try:
            # ラベル4点が未設定の場合、自動検出を試みる
            if self.label_corners is None:
                detected_corners = self.auto_detect_label_corners(frame)
                if detected_corners is not None:
                    self.label_corners = detected_corners
                    logger.info("Label corners auto-detected successfully")
            
            # ラベル4点が指定されていれば正面化ワープ
            if self.label_corners is not None and len(self.label_corners) == 4:
                warped = self.warp_to_label(frame, self.label_corners)
                self.frames.append(warped)
            else:
                self.frames.append(frame)
            if len(self.frames) == 1:
                self.camera_tracker.initialize_camera(frame.shape[:2][::-1])
                if self.mosaic_mode:
                    self.mosaic_canvas = np.zeros((self.mosaic_size[1], self.mosaic_size[0], 3), dtype=np.uint8)
                    self.mosaic_transform = np.eye(3, dtype=np.float32)
                    offset_mat = np.array([[1, 0, self.mosaic_offset[0]], [0, 1, self.mosaic_offset[1]], [0, 0, 1]])
                    warped = cv2.warpPerspective(frame, offset_mat, self.mosaic_size)
                    mask = (warped > 0)
                    self.mosaic_canvas[mask] = warped[mask]
            camera_pose = self.camera_tracker.track_camera(frame)
            self.camera_poses.append(camera_pose)
            if self.mosaic_mode and len(self.frames) > 1:
                prev_frame = self.frames[-2]
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(prev_frame, None)
                kp2, des2 = sift.detectAndCompute(frame, None)
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                if len(good) > 4:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    if H is not None:
                        H = H.astype(np.float32)
                        self.mosaic_transform = (self.mosaic_transform @ H).astype(np.float32)
                        warped = cv2.warpPerspective(frame, self.mosaic_transform, self.mosaic_size)
                        mask = (warped > 0)
                        self.mosaic_canvas[mask] = warped[mask]
            keypoints, descriptors, essential_matrix = self.object_tracker.detect_object(frame)
            if essential_matrix is not None:
                try:
                    _, R, t, _ = cv2.recoverPose(essential_matrix, 
                                               np.array([self.object_tracker.tracked_points[0]]),
                                               np.array([self.object_tracker.tracked_points[-1]]))
                    object_pose = np.eye(4)
                    object_pose[:3, :3] = R
                    object_pose[:3, 3] = t.ravel()
                    self.object_poses.append(object_pose)
                except Exception as e:
                    logger.warning(f"Error processing essential matrix: {str(e)}")
            if len(self.frames) % 100 == 0:
                logger.info(f"Processed {len(self.frames)} frames")
        except Exception as e:
            logger.error(f"Error in add_frame: {str(e)}")
        
    def reconstruct(self) -> np.ndarray:
        try:
            from ultralytics import YOLO
            import torch
            import cv2
            import numpy as np
            model = YOLO('yolov8n-seg.pt')
            # 直近10枚のみ使用
            frames_to_use = self.frames[-10:] if len(self.frames) > 10 else self.frames
            if len(frames_to_use) == 0:
                logger.warning("No frames to reconstruct")
                return None
            base_frame = frames_to_use[0]
            h, w = base_frame.shape[:2]
            aligned_objs = []
            aligned_masks = []
            weights = []

            # 最初のフレームのマスクを取得
            base_results = model(base_frame)
            if not (hasattr(base_results[0], 'masks') and base_results[0].masks is not None):
                logger.warning("No mask detected in base frame")
                return base_frame
            base_mask_data = base_results[0].masks.data.cpu().numpy()
            if base_mask_data.shape[0] == 0:
                logger.warning("Empty mask in base frame")
                return base_frame
            base_areas = base_mask_data.sum(axis=(1,2))
            base_idx = base_areas.argmax()
            base_mask = (base_mask_data[base_idx] * 255).astype(np.uint8)
            base_mask = cv2.resize(base_mask, (base_frame.shape[1], base_frame.shape[0]))
            base_mask_area = np.sum(base_mask > 0)

            # 最初のフレームはそのまま
            aligned_objs.append(cv2.bitwise_and(base_frame, base_frame, mask=base_mask))
            aligned_masks.append(base_mask)
            weights.append(base_mask.astype(np.float32) / 255.0)

            # 他のフレームのマスクを取得し、重なりを確認
            for frame in frames_to_use[1:]:
                results = model(frame)
                if not (hasattr(results[0], 'masks') and results[0].masks is not None):
                    continue
                mask_data = results[0].masks.data.cpu().numpy()
                if mask_data.shape[0] == 0:
                    continue
                areas = mask_data.sum(axis=(1,2))
                idx = areas.argmax()
                mask = (mask_data[idx] * 255).astype(np.uint8)
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                # マスクの重なりを計算
                overlap = cv2.bitwise_and(base_mask, mask)
                overlap_area = np.sum(overlap > 0)
                union_area = base_mask_area + np.sum(mask > 0) - overlap_area
                iou = overlap_area / union_area if union_area > 0 else 0
                if iou > 0.3:
                    aligned_objs.append(cv2.bitwise_and(frame, frame, mask=mask))
                    aligned_masks.append(mask)
                    weights.append(mask.astype(np.float32) / 255.0)
                    logger.debug(f"Frame added with IoU: {iou:.3f}")
                else:
                    logger.debug(f"Frame skipped due to low IoU: {iou:.3f}")

            if len(aligned_objs) < 2:
                logger.warning("Not enough frames with sufficient mask overlap")
                return base_frame

            stack = np.stack(aligned_objs, axis=0)
            mask_stack = np.stack(aligned_masks, axis=0)
            weight_stack = np.stack(weights, axis=0)

            # 合成方法を選択
            if self.blend_mode == "median":
                blend_func = np.median
                result = base_frame.copy()
                for c in range(3):
                    channel_stack = stack[..., c]
                    blended_obj = blend_func(np.where(mask_stack > 0, channel_stack, np.nan), axis=0)
                    result[..., c] = np.where(np.isnan(blended_obj), base_frame[..., c], blended_obj).astype(np.uint8)
            elif self.blend_mode == "max":
                blend_func = np.nanmax
                result = base_frame.copy()
                for c in range(3):
                    channel_stack = stack[..., c]
                    blended_obj = blend_func(np.where(mask_stack > 0, channel_stack, np.nan), axis=0)
                    result[..., c] = np.where(np.isnan(blended_obj), base_frame[..., c], blended_obj).astype(np.uint8)
            elif self.blend_mode == "min":
                blend_func = np.nanmin
                result = base_frame.copy()
                for c in range(3):
                    channel_stack = stack[..., c]
                    blended_obj = blend_func(np.where(mask_stack > 0, channel_stack, np.nan), axis=0)
                    result[..., c] = np.where(np.isnan(blended_obj), base_frame[..., c], blended_obj).astype(np.uint8)
            elif self.blend_mode == "weighted":
                # 重み付き平均
                result = base_frame.copy()
                total_weight = np.sum(weight_stack, axis=0)
                for c in range(3):
                    channel_stack = stack[..., c]
                    weighted_sum = np.nansum(channel_stack * weight_stack, axis=0)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        blended_obj = weighted_sum / total_weight
                    result[..., c] = np.where(total_weight > 0, blended_obj, base_frame[..., c]).astype(np.uint8)
            else:
                # デフォルトはmedian
                blend_func = np.median
                result = base_frame.copy()
                for c in range(3):
                    channel_stack = stack[..., c]
                    blended_obj = blend_func(np.where(mask_stack > 0, channel_stack, np.nan), axis=0)
                    result[..., c] = np.where(np.isnan(blended_obj), base_frame[..., c], blended_obj).astype(np.uint8)

            # --- アンシャープマスクでシャープ化 ---
            blur = cv2.GaussianBlur(result, (0, 0), 3)
            sharpened = cv2.addWeighted(result, 1.5, blur, -0.5, 0)
            # --- 緑色マスク混入防止処理 ---
            green_mask = (sharpened[..., 1] > 180) & (sharpened[..., 1] > sharpened[..., 0] + 40) & (sharpened[..., 1] > sharpened[..., 2] + 40)
            for c in range(3):
                sharpened[..., c][green_mask] = base_frame[..., c][green_mask]
            logger.info(f"Returning YOLOv8-seg {self.blend_mode} blended + sharpened image. shape={sharpened.shape}")

            # --- 超解像AI（Real-ESRGAN, Python API）でアップスケール ---
            try:
                import torch
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                from PIL import Image
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                upsampler = RealESRGANer(
                    scale=2,
                    model_path='weights/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=True if torch.cuda.is_available() else False,
                    device=device
                )
                img_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                sr_img, _ = upsampler.enhance(np.array(pil_img), outscale=2)
                sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
                logger.info(f"Super-resolution applied. shape={sr_img_bgr.shape}")
                return sr_img_bgr
            except Exception as e:
                logger.error(f"Super-resolution error: {str(e)}")
                return sharpened
        except Exception as e:
            logger.error(f"Error in YOLOv8-seg blending: {str(e)}")
        # --- モザイク合成 ---
        if self.mosaic_mode and self.mosaic_canvas is not None:
            logger.info(f"Returning mosaic image. shape={self.mosaic_canvas.shape}")
            return self.mosaic_canvas
        # --- 既存の3D再構成処理 ---
        try:
            if len(self.frames) < 2:
                logger.warning("Not enough frames for reconstruction")
                return None
            if len(self.object_poses) > 0:
                poses = torch.tensor(np.array(self.object_poses), dtype=torch.float32)
            else:
                poses = torch.tensor(np.array(self.camera_poses), dtype=torch.float32)
            logger.debug(f"Using poses tensor shape: {poses.shape}")
            with torch.no_grad():
                rendered_images = self.gaussian_model(poses)
            logger.info(f"reconstruct: rendered_images shape={rendered_images.shape}")
            if rendered_images.ndim == 4:
                final_image = rendered_images[-1].cpu().numpy()
            elif rendered_images.ndim == 3:
                final_image = rendered_images.cpu().numpy()
            else:
                raise ValueError(f"Unexpected rendered_images shape: {rendered_images.shape}")
            if final_image.max() > 0:
                final_image = final_image / final_image.max()
            final_image = (final_image * 255).clip(0, 255).astype(np.uint8)
            if final_image.shape != (720, 1280, 3):
                final_image = np.reshape(final_image, (720, 1280, 3))
            logger.info(f"Reconstruction completed. shape={final_image.shape}, dtype={final_image.dtype}")
            return final_image
        except Exception as e:
            logger.error(f"Error in reconstruct: {str(e)}")
            return None 