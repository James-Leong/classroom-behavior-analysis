import io
import logging
import pickle

from contextlib import redirect_stdout, redirect_stderr, contextmanager
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import warnings
import os

from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis
from ultralytics import YOLO

# 忽略 albumentations 检查版本时网络超时的无害警告
warnings.filterwarnings(
    "ignore",
    message="Error fetching version info",
    category=UserWarning,
    module=r"albumentations.*",
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 常见系统字体候选（macOS/Windows/Linux），按需扩展
FONT_LIST = [
    # macOS
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/AppleGothic.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/STHeiti.ttf",
    # Windows (注意字符串中的反斜杠已转义)
    "C:\\Windows\\Fonts\\msyh.ttc",
    "C:\\Windows\\Fonts\\msyh.ttf",
    "C:\\Windows\\Fonts\\simsun.ttc",
    "C:\\Windows\\Fonts\\arialuni.ttf",
    # 常见 Linux 字体
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def draw_text_cn(img: np.ndarray, text: str, org: Tuple[int, int], font_size: int = 14, color=(255, 255, 255)):
    """
    在 OpenCV 图像上绘制中文或其他 Unicode 文本（优先使用 PIL），若失败则回退到 cv2.putText。

    Args:
        img: BGR 格式的 numpy 图像
        text: 要绘制的文本（支持中文）
        org: 文本左上角位置 (x, y)
        font_size: 字体大小（像素）
        color: BGR 颜色元组
    """
    try:
        # 转为 PIL (RGB)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        font = None
        for p in FONT_LIST:
            try:
                font = ImageFont.truetype(p, font_size)
                break
            except Exception:
                continue

        if font is None:
            font = ImageFont.load_default()

        # PIL 使用 RGB，color 为 BGR，需转换
        rgb_color = (color[2], color[1], color[0])
        draw.text(org, text, font=font, fill=rgb_color)

        # 写回到 numpy BGR 图像（原地替换内容）
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    except Exception:
        # 回退：使用 OpenCV 的 putText（可能中文仍然乱码）
        # 这里尽力保持与 PIL 相近的大小映射
        font_scale = max(0.3, font_size / 24.0)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (color[0], color[1], color[2]), 1, cv2.LINE_AA)


def measure_text_cn(text: str, font_size: int = 14) -> Tuple[int, int]:
    """使用 PIL 测量文本像素尺寸，回退到 OpenCV 测量。"""
    try:

        font = None
        for p in FONT_LIST:
            try:
                font = ImageFont.truetype(p, font_size)
                break
            except Exception:
                continue

        if font is None:
            font = ImageFont.load_default()

        # 使用 textbbox 得到精确尺寸
        dummy = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return int(width), int(height)

    except Exception:
        # 回退：OpenCV 近似测量
        font_scale = max(0.3, font_size / 24.0)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        return int(w), int(h)


@contextmanager
def _suppress_fds():
    """Context manager that redirects FD 1 and 2 to /dev/null.

    This suppresses output from C-level prints and other threads that bypass
    Python's sys.stdout/sys.stderr objects.
    """
    devnull = os.open(os.devnull, os.O_RDWR)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)


class FaceRecognizer:
    """
    人脸识别器，基于Yolov11和InsightFace模型实现

    主要功能：
    1. 基于yolo的人脸检测
    2. 人脸图像的自动切割与调整
    3. 基于insightface的人脸识别
    """

    def __init__(
        self,
        detection_model: str = 'yolo11n.pt',
        recognition_model: str = 'buffalo_l',
        gallery_path: str = 'gallery',
        threshold: float = 0.4,
        quality_threshold: float = 0.7,
        det_size: int = 320,
        device: str = 'auto',
        rebuild_gallery: bool = False,
        yolo_conf: float = 0.12,
    ):
        """
        初始化人脸识别器

        Args:
            detection_model: YOLO检测模型路径或名称，默认'yolo11n.pt'
            recognition_model: InsightFace模型名称，默认'buffalo_l'
            gallery_path: 图库路径，包含按人员命名的子目录，每个子目录包含该人员的人脸图像
            threshold: 匹配阈值，默认0.4
            quality_threshold: 人脸质量阈值，默认0.7
            device: 计算设备，'auto'/'cpu'/'gpu'
            rebuild_gallery: 是否强制重建图库
        """
        self.detection_model = detection_model
        self.recognition_model = recognition_model
        self.gallery_path = Path(gallery_path)
        self.threshold = threshold
        self.quality_threshold = quality_threshold
        self.det_size: Tuple[int, int] = (det_size, det_size,)  # 检测尺寸配置
        self.device = device
        self.rebuild_gallery = rebuild_gallery
        # YOLO 检测置信度阈值（可在外部传入以提高召回）
        self.yolo_conf = float(yolo_conf)
        # YOLO 推理输入尺寸（单边像素），增大以提升对小目标的检测（推荐）
        self.yolo_imgsz = 960
        # 是否启用平铺检测（对教室这种密集小人脸场景有帮助，推荐启用）
        self.yolo_use_tiling = True
        # 平铺大小与重叠（像素），推荐 tile_size=800, overlap=0.25
        self.yolo_tile_size = 800
        self.yolo_tile_overlap = 0.25
        # 合并重复框的 IoU 阈值（NMS），推荐 0.25
        self.yolo_nms_thresh = 0.25

        # 模型实例
        self._detect_app: Optional[YOLO] = None
        self._recogn_app: Optional[FaceAnalysis] = None

        # 图库embeddings
        self.gallery_embeddings = {}
        self.gallery_stats = {}  # 存储质量统计信息

        # 运行参数
        self.ctx_id = -1  # -1表示CPU，0表示第一个GPU

        # 初始化模型
        self._initialize_models()
        self._load_or_build_gallery()
    
    def _initialize_models(self):
        """初始化InsightFace模型"""
        try:
            # 选择设备
            if self.device == "auto":
                device_cnt = torch.cuda.device_count()
                device = "gpu" if device_cnt > 0 else "cpu"
            else:
                device = self.device

            # 配置providers，使用当前空闲的GPU
            if device == "gpu":
                providers = ["CUDAExecutionProvider"]
                self.ctx_id = 0  # 选择第一个GPU
            else:
                providers = ["CPUExecutionProvider"]
                self.ctx_id = -1

            # 在创建模型时也抑制其可能的底层输出
            with _suppress_fds():
                self._detect_app = YOLO(self.detection_model)
            logger.info(f"已加载 YOLO 模型: {self.detection_model}")

            with _suppress_fds():
                self._recogn_app = FaceAnalysis(
                    name=self.recognition_model,
                    providers=providers,
                    allowed_modules=["detection", "recognition"],
                )
            logger.info(f"已加载 InsightFace 模型: {self.recognition_model}")

            # 使用统一的 det_size 进行 prepare
            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(buf):
                self._recogn_app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise

    def _build_gallery(self):
        """构建图库"""
        logger.info("开始构建图库...")

        total_images = 0
        successful_images = 0
        
        for person_dir in self.gallery_path.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            # 采用不区分大小写的后缀匹配，避免漏掉像 0001.JPG 这种大写扩展名
            image_files = [p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]

            logger.info(f"处理 {person_name}: {len(image_files)} 张图像")
            total_images += len(image_files)
            
            person_embeddings = []
            person_qualities = []
            
            for img_file in image_files:
                try:
                    image = cv2.imread(str(img_file))
                    if image is None:
                        logger.warning(f"无法读取图像: {img_file}")
                        continue
                    
                    # 鲁棒人脸检测
                    faces = self.detect_faces(image)
                    
                    if len(faces) == 0:
                        logger.warning(f"在 {img_file} 中未检测到人脸")
                        continue
                    
                    # 选择质量最高的人脸
                    best_face = None
                    best_quality = 0
                    
                    for face in faces:
                        quality = self.assess_face_quality(face, image.shape)
                        if quality > best_quality:
                            best_quality = quality
                            best_face = face
                    
                    if best_face and best_quality >= self.quality_threshold:
                        embedding = best_face.embedding
                        person_embeddings.append(embedding)
                        person_qualities.append(best_quality)
                        
                        logger.info(f"  {img_file.name}: 质量{best_quality:.3f}, 尺寸{best_face.det_size}")
                        successful_images += 1
                    else:
                        logger.warning(f"  {img_file.name}: 人脸质量过低 ({best_quality:.3f}), faces={len(faces)}")
                        
                except Exception as e:
                    logger.error(f"处理 {img_file} 失败: {e}")
            
            if person_embeddings:
                embeddings = np.array(person_embeddings)
                qualities = np.array(person_qualities)
                
                # 质量加权平均
                weights = qualities / qualities.sum()
                avg_embedding = np.average(embeddings, axis=0, weights=weights)
                
                self.gallery_embeddings[person_name] = avg_embedding
                self.gallery_stats[person_name] = {
                    'quality': np.mean(qualities),
                    'count': len(person_embeddings),
                    'avg_norm': np.linalg.norm(avg_embedding)
                }
                
                logger.info(f"  ✅ 成功添加: {len(person_embeddings)} 个高质量embeddings")
            else:
                logger.warning(f"  ❌ {person_name} 没有合格的人脸图像")
        
        # 保存图库
        if self.gallery_embeddings:
            gallery_data = {
                'embeddings': self.gallery_embeddings,
                'stats': self.gallery_stats,
                'threshold': self.threshold,
                'quality_threshold': self.quality_threshold
            }
            
            embeddings_file = self.gallery_path / 'gallery_embeddings.pkl'
            with open(embeddings_file, 'wb') as f:
                pickle.dump(gallery_data, f)
            
            logger.info(f"图库构建完成: {len(self.gallery_embeddings)} 个人, {successful_images}/{total_images} 张图像")

    def _crop_and_resize(self, crop: np.ndarray, target: Tuple[int,int]) -> Tuple[np.ndarray, float, float]:
        """
        把 crop 缩放到 target (w,h)，返回 (resized, sx, sy) 其中 sx = orig_w / resized_w 用于反向映射。
        
        Args:
            crop: 裁剪图像
            target: 目标尺寸 (w, h)
        
        Returns:
            resized: 缩放后的图像
            sx: 宽度缩放比例
            sy: 高度缩放比例
        """
        tw, th = target
        try:
            resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
        except Exception:
            resized = crop.copy()
        # 缩放比例（用于将 face.bbox 映回原图时使用）
        sx = float(crop.shape[1]) / float(resized.shape[1]) if resized.shape[1] > 0 else 1.0
        sy = float(crop.shape[0]) / float(resized.shape[0]) if resized.shape[0] > 0 else 1.0
        return resized, sx, sy

    def _detect_with_yolo(self, image: np.ndarray, conf: float = None) -> List[Tuple[int,int,int,int]]:
        """
        使用 YOLO 检测人脸，返回 xyxy 列表（整数）
        
        Args:
            image: 需要检测的图片
            conf: 置信度
        
        Returns:
            boxes: 识别的图片坐标
        """
        boxes = []
        if self._detect_app is None:
            return boxes

        try:
            # 支持从实例默认值覆盖阈值
            if conf is None:
                conf = float(self.yolo_conf)
            # 使用更大的推理尺寸以提升对小人脸的检测能力
            with _suppress_fds():
                try:
                    results = self._detect_app(image, imgsz=self.yolo_imgsz)
                except TypeError:
                    # 兼容不同 ultralytics 版本的参数签名
                    results = self._detect_app(image)

            if not results:
                return boxes

            r = results[0]
            # 仅支持现代返回格式（.boxes 每项包含 .xyxy 和 .conf），简化兼容逻辑
            for box in r.boxes:
                xyxy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, 'cpu') else np.array(box.xyxy)
                if xyxy.ndim == 2:
                    x1, y1, x2, y2 = map(int, xyxy[0])
                else:
                    x1, y1, x2, y2 = map(int, xyxy)
                score = float(box.conf)
                if score >= conf:
                    boxes.append((x1, y1, x2, y2))

        except Exception as e:
            logger.warning(f"YOLO 检测失败: {e}")

        return boxes

    def _detect_with_yolo_batch(self, images: List[np.ndarray], conf: float = None) -> List[List[Tuple[int, int, int, int]]]:
        """使用 YOLO 对一批图像进行检测，返回每张图像的 bbox 列表。

        Args:
            images: BGR 图像列表
            conf: 置信度阈值

        Returns:
            每张图像对应的 bbox 列表，元素形如 [(x1, y1, x2, y2), ...]
        """
        if self._detect_app is None or not images:
            return [[] for _ in images]

        boxes_batch: List[List[Tuple[int, int, int, int]]] = [[] for _ in images]
        try:
            # 支持从实例默认值覆盖阈值
            if conf is None:
                conf = float(self.yolo_conf)
            # 使用 FD 级别抑制，能覆盖多线程 / C 层打印
            with _suppress_fds():
                results = self._detect_app(images)
            # Ultralytics YOLO 可能返回单个或列表，这里统一转换为列表
            if not isinstance(results, (list, tuple)):
                results = [results]

            for idx, (img, r) in enumerate(zip(images, results)):
                cur_boxes: List[Tuple[int, int, int, int]] = []
                try:
                        if r is None or not hasattr(r, 'boxes') or r.boxes is None:
                            boxes_batch[idx] = []
                            continue
                        for box in r.boxes:
                            xyxy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, 'cpu') else np.array(box.xyxy)
                            if xyxy.ndim == 2:
                                x1, y1, x2, y2 = map(int, xyxy[0])
                            else:
                                x1, y1, x2, y2 = map(int, xyxy)
                            score = float(box.conf)
                            if score >= conf:
                                cur_boxes.append((x1, y1, x2, y2))
                except Exception as e:
                    logger.warning(f"YOLO 批量检测失败: index={idx}, error={e}")
                boxes_batch[idx] = cur_boxes

        except Exception as e:
            logger.warning(f"YOLO 批量检测调用失败: {e}")
            return [[] for _ in images]

        return boxes_batch

    def _boxes_nms(self, boxes_with_scores: List[Tuple[int, int, int, int, float]], iou_thresh: float) -> List[Tuple[int, int, int, int]]:
        """简单 NMS：按 score 降序，保留与当前保留框 IoU < thresh 的框。"""
        if not boxes_with_scores:
            return []
        boxes_with_scores = sorted(boxes_with_scores, key=lambda x: x[4], reverse=True)
        keep: List[Tuple[int, int, int, int]] = []
        for b in boxes_with_scores:
            x1, y1, x2, y2, s = b
            discard = False
            for k in keep:
                # 计算 IoU
                xx1 = max(x1, k[0])
                yy1 = max(y1, k[1])
                xx2 = min(x2, k[2])
                yy2 = min(y2, k[3])
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                area_a = max(0, x2 - x1) * max(0, y2 - y1)
                area_b = max(0, k[2] - k[0]) * max(0, k[3] - k[1])
                denom = area_a + area_b - inter
                iou = inter / denom if denom > 0 else 0.0
                if iou >= iou_thresh:
                    discard = True
                    break
            if not discard:
                keep.append((x1, y1, x2, y2))
        return keep

    def _detect_with_yolo_tiled(self, image: np.ndarray, conf: float = None) -> List[Tuple[int, int, int, int]]:
        """对大图启用平铺检测，提升对小人脸的召回。

        返回 xyxy 框列表，与 `_detect_with_yolo` 保持一致。
        """
        if self._detect_app is None:
            return []
        if conf is None:
            conf = float(self.yolo_conf)

        h, w = image.shape[:2]
        tile = int(self.yolo_tile_size)
        overlap = float(self.yolo_tile_overlap)
        step = max(1, int(tile * (1.0 - overlap)))

        collected: List[Tuple[int, int, int, int, float]] = []

        xs = list(range(0, w, step))
        ys = list(range(0, h, step))
        for yy in ys:
            for xx in xs:
                x1 = xx
                y1 = yy
                x2 = min(w, xx + tile)
                y2 = min(h, yy + tile)
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                try:
                    # 与单图相同的 imgsz 参数尝试（使用 FD 级别抑制，以屏蔽 ultralytics 输出）
                    with _suppress_fds():
                        try:
                            results = self._detect_app(crop, imgsz=self.yolo_imgsz)
                        except TypeError:
                            results = self._detect_app(crop)
                    if not results:
                        continue
                    r = results[0]
                    if r is None or not hasattr(r, 'boxes') or r.boxes is None:
                        continue
                    for box in r.boxes:
                        xyxy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, 'cpu') else np.array(box.xyxy)
                        if xyxy.ndim == 2:
                            bx1, by1, bx2, by2 = map(int, xyxy[0])
                        else:
                            bx1, by1, bx2, by2 = map(int, xyxy)
                        score = float(box.conf)
                        if score >= conf:
                            # 转回原图坐标
                            gx1 = x1 + bx1
                            gy1 = y1 + by1
                            gx2 = x1 + bx2
                            gy2 = y1 + by2
                            collected.append((gx1, gy1, gx2, gy2, score))
                except Exception:
                    continue

        # NMS 合并重复框
        boxes = self._boxes_nms(collected, iou_thresh=self.yolo_nms_thresh)

        # 简洁汇总日志：平铺数、收集到的候选框和 NMS 后框数（仅在调试级别输出）
        try:
            tiles_processed = len(xs) * len(ys)
            logger.debug(f"YOLO tiled: tiles={tiles_processed}, collected={len(collected)}, nms_kept={len(boxes)}")
        except Exception:
            pass

        return boxes

    def _detect_faces_with_boxes(self, image: np.ndarray, yolo_boxes: List[Tuple[int, int, int, int]]) -> List:
        """在已给定 YOLO bbox 的前提下，执行裁剪 + InsightFace 识别并返回人脸列表。"""
        all_faces = []
        img_h, img_w = image.shape[:2]

        if not yolo_boxes:
            # 若 YOLO 未给出候选框，则回退到 InsightFace 对整图检测（可检测小/密集人脸）
            try:
                faces = self._recogn_app.get(image)
                if not faces:
                    return []
                for face in faces:
                    try:
                        # face.bbox 已为原图坐标
                        face.det_size = (img_w, img_h)
                        face.enhancement = 'full_image'
                        all_faces.append(face)
                    except Exception:
                        continue
                logger.info(f"InsightFace 整图检测到 {len(all_faces)} 张人脸（回退路径）")
                merged = self._merge_close_faces(all_faces, image.shape, center_tol=0.012)
                return merged
            except Exception as e:
                logger.debug(f"InsightFace 整图回退失败: {e}")
                return []

        import time
        st = time.time()

        for (bx1, by1, bx2, by2) in yolo_boxes:
            bw = bx2 - bx1
            bh = by2 - by1
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)

            x1 = max(0, bx1 - pad_x)
            y1 = max(0, by1 - pad_y)
            x2 = min(img_w, bx2 + pad_x)
            y2 = min(img_h, by2 + pad_y)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            resized, sx, sy = self._crop_and_resize(crop, self.det_size)
            target_det = self.det_size

            try:
                faces = self._recogn_app.get(resized)
            except Exception as e:
                logger.debug(f"InsightFace 对裁剪图像处理失败: {e}")
                continue

            for face in faces:
                try:
                    bbox = face.bbox.astype(float)
                    ox1 = int(bbox[0] * sx) + x1
                    oy1 = int(bbox[1] * sy) + y1
                    ox2 = int(bbox[2] * sx) + x1
                    oy2 = int(bbox[3] * sy) + y1

                    face.bbox = np.array([ox1, oy1, ox2, oy2])

                    if hasattr(face, 'kps') and face.kps is not None:
                        kps = np.asarray(face.kps).astype(float).reshape(-1, 2)
                        kps[:, 0] = kps[:, 0] * sx + x1
                        kps[:, 1] = kps[:, 1] * sy + y1
                        face.kps = kps

                    face.det_size = target_det
                    face.enhancement = 'unified_crop'

                    all_faces.append(face)
                except Exception:
                    continue

        ed = time.time()
        logger.info(f"人脸识别耗时: {ed - st:.2f} 秒，检测到 {len(all_faces)} 张人脸")

        try:
            merged = self._merge_close_faces(all_faces, image.shape, center_tol=0.03)
            return merged
        except Exception:
            return all_faces

    def _draw(self, image: np.ndarray, result: Dict, person: int = None):
        """
        结果绘制
        
        Args:
            image: 需要绘制的图片
            result: 识别结果字典
            person: 可选，人员索引
        """
        x1, y1, x2, y2 = result['bbox']
        identity = result['identity']
        similarity = result['similarity']
        quality = result['quality']
        
        # 根据身份和质量选择颜色
        if identity == "未知":
            color = (0, 0, 255)  # 红色
        elif quality >= 0.8:
            color = (0, 255, 0)  # 绿色（高质量）
        elif quality >= 0.6:
            color = (0, 255, 255)  # 黄色（中等质量）
        else:
            color = (0, 165, 255)  # 橙色（低质量）
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制关键点（如果有）
        if result['landmarks'] is not None:
            landmarks = result['landmarks'].astype(int)
            for (x, y) in landmarks:
                cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
        
        # 准备标签文本（可能含中文）
        label = f"{identity} ({similarity:.3f})"
        quality_label = f"Q:{quality:.2f}"

        # 基于人脸 bbox 高度计算字体大小（相对比例），并限制在合理范围
        face_h = max(12, y2 - y1)
        # 标签字体取人脸高度的 18%，质量字体取 14%
        label_font_size = max(12, int(face_h * 0.12))
        quality_font_size = max(10, int(face_h * 0.08))

        # 测量文本尺寸以绘制背景
        text_w, text_h = measure_text_cn(label, label_font_size)
        padding_x = max(6, int(label_font_size * 0.3))
        padding_y = max(4, int(label_font_size * 0.2))

        bg_x1 = x1
        bg_y1 = max(0, y1 - text_h - padding_y * 2)
        bg_x2 = x1 + text_w + padding_x * 2
        bg_y2 = y1

        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

        # 在背景内绘制标签（使用内边距）
        text_org = (bg_x1 + padding_x, bg_y1 + padding_y)
        draw_text_cn(image, label, text_org, font_size=label_font_size, color=(255, 255, 255))

        # 绘制质量信息（放在 bbox 底部）
        q_w, q_h = measure_text_cn(quality_label, quality_font_size)
        q_org = (x1, min(image.shape[0] - q_h - 2, y1 + 15))
        draw_text_cn(image, quality_label, q_org, font_size=quality_font_size, color=(255, 255, 255))

    def _load_or_build_gallery(self):
        """加载或构建图库"""
        embeddings_file = self.gallery_path / 'gallery_embeddings.pkl'
        
        if embeddings_file.exists() and not self.rebuild_gallery:
            try:
                with open(embeddings_file, 'rb') as f:
                    gallery_data = pickle.load(f)
                
                self.gallery_embeddings = gallery_data['embeddings']
                self.gallery_stats = gallery_data.get('stats', {})
                
                logger.info(f"已加载图库: {len(self.gallery_embeddings)} 个人")
                # self.analyze_gallery_quality()
                
            except Exception as e:
                logger.warning(f"加载图库失败: {e}")
                self._build_gallery()
        else:
            self._build_gallery()

    def _merge_close_faces(self, faces: List, image_shape: Tuple[int,int], center_tol: float = 0.012) -> List:
        """
        合并在图像中位置非常接近的候选人脸，避免重复绘制。

        Args:
            faces: InsightFace 返回的 face 对象列表（含 bbox）
            image_shape: 原图 shape (h, w, ...)
            center_tol: 中心点距离阈值，按最大边长的比例计算，若两框中心距离小于 tol*max_dim 则合并

        Returns:
            faces: 合并后的 face 列表，保留每组中质量最高的 face（使用 assess_face_quality 评分）。
        """
        if not faces or len(faces) <= 1:
            return faces

        h, w = image_shape[:2]
        max_dim = max(w, h)
        tol = center_tol * max_dim

        centers = []
        for f in faces:
            try:
                bx = f.bbox.astype(float)
                cx = (bx[0] + bx[2]) / 2.0
                cy = (bx[1] + bx[3]) / 2.0
            except Exception:
                # 兜底
                cx, cy = 0.0, 0.0
            centers.append((cx, cy))

        used = [False] * len(faces)
        merged = []

        for i, fi in enumerate(faces):
            if used[i]:
                continue
            group = [i]
            used[i] = True
            for j in range(i + 1, len(faces)):
                if used[j]:
                    continue
                dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                if dist <= tol:
                    group.append(j)
                    used[j] = True

            # 从 group 中选品质最高的 face 保留
            best_idx = group[0]
            try:
                best_score = self.assess_face_quality(faces[best_idx], image_shape)
            except Exception:
                best_score = getattr(faces[best_idx], 'det_score', 0.5)

            for idx in group[1:]:
                try:
                    score = self.assess_face_quality(faces[idx], image_shape)
                except Exception:
                    score = getattr(faces[idx], 'det_score', 0.5)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            merged.append(faces[best_idx])

        return merged

    def analyze_gallery_quality(self):
        """
        分析图库质量。
        输出图库中每个人的质量统计信息，包括平均质量、样本数量和Embedding范数。
        """
        if not self.gallery_stats:
            return

        logger.info("=== 图库质量分析 ===")

        for person_name, stats in self.gallery_stats.items():
            logger.info(f"{person_name}:")
            logger.info(f"  平均质量: {stats['quality']:.3f}")
            logger.info(f"  样本数量: {stats['count']}")
            logger.info(f"  Embedding范数: {stats['avg_norm']:.3f}")

        # 计算类间相似度
        if len(self.gallery_embeddings) >= 2:
            logger.info("类间相似度分析:")
            names = list(self.gallery_embeddings.keys())
            
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    emb1 = self.gallery_embeddings[names[i]]
                    emb2 = self.gallery_embeddings[names[j]]
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    logger.info(f"  {names[i]} vs {names[j]}: {similarity:.4f}")

    def assess_face_quality(self, face, image_shape: Tuple[int, int]) -> float:
        """
        评估人脸质量

        Args:
            face: 检测到的人脸
            image_shape: 图像尺寸 (height, width)

        Returns:
            quality_score: 人脸质量分数 (0-1)
        """
        quality_score = 0.5  # 基础分数

        try:
            # 1. 人脸尺寸评估
            bbox = face.bbox
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            
            # 计算相对尺寸
            img_height, img_width = image_shape[:2]
            relative_size = (face_width * face_height) / (img_width * img_height)
            
            # 课堂场景：远距小脸是常态，简化尺寸评估
            if relative_size >= 0.02:
                quality_score += 0.15
            else:
                quality_score -= 0.05  # 仅轻微惩罚极小人脸
            
            # 2. 检测置信度
            det_score = getattr(face, 'det_score', 0.8)
            quality_score += (det_score - 0.5) * 0.3
            
            # 3. 姿态评估（如果有关键点信息）
            if hasattr(face, 'kps') and face.kps is not None:
                kps = face.kps
                # 简单的姿态评估：检查眼睛位置
                left_eye, right_eye = kps[0], kps[1]
                eye_distance = np.linalg.norm(right_eye - left_eye)
                
                # 理想的眼睛距离
                if eye_distance > face_width * 0.3:
                    quality_score += 0.1
            
            # 确保分数在合理范围
            quality_score = max(0.1, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"质量评估失败: {e}")
            quality_score = 0.5
        
        return quality_score

    def detect_faces(self, image: np.ndarray) -> List:
        """
        从图像中检测人脸

        Args:
            image: 输入图像

        Returns:
            all_faces: 检测到的所有人脸列表
        """
        # 按照 YOLO -> 裁剪 -> 统一缩放 -> InsightFace 的流程执行人脸识别
        if self.yolo_use_tiling:
            yolo_boxes = self._detect_with_yolo_tiled(image)
        else:
            yolo_boxes = self._detect_with_yolo(image)
        logger.debug(f"YOLO 检测到 {len(yolo_boxes)} 个候选框，使用统一流程进行识别")
        return self._detect_faces_with_boxes(image, yolo_boxes)

    def detect_faces_batch(self, images: List[np.ndarray]) -> List[List]:
        """对多张图像执行人脸检测与识别，尽可能在 YOLO 侧做 batch，以提高 GPU 利用率。

        Args:
            images: BGR 图像列表

        Returns:
            每张图像对应的人脸列表（与 detect_faces 单图接口兼容）
        """
        if not images:
            return []

        faces_batch: List[List] = []
        if self.yolo_use_tiling:
            for img in images:
                yolo_boxes = self._detect_with_yolo_tiled(img)
                faces = self._detect_faces_with_boxes(img, yolo_boxes)
                faces_batch.append(faces)
            return faces_batch

        yolo_boxes_batch = self._detect_with_yolo_batch(images)
        for img, yolo_boxes in zip(images, yolo_boxes_batch):
            faces = self._detect_faces_with_boxes(img, yolo_boxes)
            faces_batch.append(faces)
        return faces_batch

    def get_gallery_info(self) -> Dict:
        """获取图库信息"""
        info = {
            'total_persons': len(self.gallery_embeddings),
            'person_names': list(self.gallery_embeddings.keys()),
            'similarity_threshold': self.threshold,
            'quality_threshold': self.quality_threshold,
            'stats': self.gallery_stats
        }
        
        return info

    def proccess(self, image_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        人脸识别
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径（可选）
            
        Returns:
            result_image: 带标注的结果图像
            recognition_results: 识别结果列表
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        result_image = image.copy()
        
        # 人脸检测
        faces = self.detect_faces(image)

        if len(faces) == 0:
            logger.warning(f"在 {image_path} 中未检测到人脸")
            if output_path:
                cv2.imwrite(output_path, result_image)
            return result_image, []
        
        recognition_results = []
        
        for i, face in enumerate(faces):
            # 评估质量
            quality = self.assess_face_quality(face, image.shape)
            
            # 获取人脸框
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # 识别身份
            identity, similarity = self.recognize_identity(face.embedding, quality)

            # 准备结果
            result = {
                'bbox': (x1, y1, x2, y2),
                'identity': identity,
                'similarity': similarity,
                'quality': quality,
                'landmarks': getattr(face, 'kps', None),
                'det_size': getattr(face, 'det_size', None),
                'enhancement': getattr(face, 'enhancement', 'original')
            }
            
            recognition_results.append(result)
            
            # 绘制结果
            self._draw(result_image, result, i)
            
            logger.info(f"检测到人脸 {i+1}: {identity} (相似度: {similarity:.4f}, 质量: {quality:.3f})")
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, result_image)
            logger.info(f"结果图像已保存至: {output_path}")
        
        return result_image, recognition_results

    def recognize_identity(self, embedding: np.ndarray, quality: float, debug: bool = False, topk: int = 5) -> Tuple[str, float]:
        """
        身份识别，考虑质量因素
        
        Args:
            embedding: 人脸embedding
            quality: 人脸质量分数
            
        Returns:
            (身份, 相似度)
        """
        if not self.gallery_embeddings:
            return "未知", 0.0
        
        best_match = "未知"
        best_similarity = 0.0

        # 计算所有相似度并排序（用于调试 / 阈值调整参考）
        sims = []
        try:
            emb_norm = np.linalg.norm(embedding)
        except Exception:
            emb_norm = 0.0

        for person_name, gallery_embedding in self.gallery_embeddings.items():
            try:
                g_norm = np.linalg.norm(gallery_embedding)
                if emb_norm == 0 or g_norm == 0:
                    similarity = 0.0
                else:
                    similarity = float(np.dot(embedding, gallery_embedding) / (emb_norm * g_norm))
            except Exception:
                similarity = 0.0
            sims.append((person_name, similarity))

        sims.sort(key=lambda x: x[1], reverse=True)

        if sims:
            best_match, best_similarity = sims[0]

        # 调试输出 top-k 相似度，便于观察相似度分布
        if debug:
            topk = max(1, int(topk))
            logger.info(f"recognize_identity debug: top{topk} -> {sims[:topk]}")

        # 检查是否超过阈值
        if best_similarity >= self.threshold:
            return best_match, best_similarity
        else:
            return "未知", best_similarity




# 使用示例和测试函数
def main():
    """主函数：演示增强版识别器的使用"""
    
    # 创建增强版识别器
    recognizer = FaceRecognizer(
        gallery_path='data/id_photos',
        threshold=0.4,  # 降低阈值以提高召回率
        quality_threshold=0.6,  # 适中的质量要求
        det_size=320,
        device='auto',
        rebuild_gallery=True,  # 强制重建图库以确保最新数据
    )

    # 测试图像
    test_images = [
        'hangzhou.jpeg'  # 测试图像
    ]
    
    logger.info("=== 人脸识别测试 ===")
    
    for img_path in test_images:
        if not Path(img_path).exists():
            continue
        
        logger.info(f"🔍 测试: {img_path}")
        
        try:
            # 识别
            result_img, results = recognizer.proccess(
                img_path,
                f'output_{Path(img_path).stem}.jpg'
            )

            if len(results) == 0:
                logger.warning("未检测到人脸")
            else:
                logger.info(f"检测到 {len(results)} 个人脸")

                for i, result in enumerate(results):
                    identity = result['identity']
                    similarity = result['similarity']
                    quality = result['quality']

                    if identity == "未知":
                        logger.info(f"人脸 {i+1}: {identity} (相似度: {similarity:.4f}, 质量: {quality:.3f})")
                    else:
                        logger.info(f"人脸 {i+1}: {identity} (相似度: {similarity:.4f}, 质量: {quality:.3f}) ✅")

        except Exception as e:
            logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    main()
