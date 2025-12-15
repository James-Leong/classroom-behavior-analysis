import io
import time

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from insightface.app import FaceAnalysis
from insightface.app.common import Face
from ultralytics import YOLO as YOLOType

from src.face.gallery import Gallery, GalleryConfig
from src.face.matcher import CosineMatcher, MatcherConfig
from src.utils.draw import draw_text_cn, draw_texts_cn, measure_text_cn
from src.utils.log import get_logger, suppress_fds
from src.utils.math import l2_normalize

logger = get_logger(__name__)

# 进程内模型缓存：显著减少重复初始化耗时（例如 pytest 多用例/多次构造 Recognizer）。
# 注意：缓存 key 需要包含会影响输出的关键参数（providers/ctx_id/det_size/model name）。
_FACEAPP_CACHE: Dict[Tuple, FaceAnalysis] = {}
_YOLO_CACHE: Dict[str, Any] = {}


class FaceRecognizer:
    """
    人脸识别器：以 InsightFace 作为主检测 + 主特征提取链路，
    YOLO 仅作为可选的“候选框提案器”。

    主要功能：
    1. InsightFace 检测 + 识别（主链路，鲁棒）
    2. （可选）YOLO 候选框提案 + InsightFace 二次确认
    3. 图库支持每人多 embedding（更稳，降低单张照片失败风险）
    """

    def __init__(
        self,
        recognition_model: str = "buffalo_l",
        gallery_path: str = "gallery",
        threshold: float = 0.4,
        quality_threshold: float = 0.7,
        det_size: int = 640,
        device: str = "auto",
        rebuild_gallery: bool = False,
        detection_model: Optional[str] = None,
        yolo_conf: float = 0.12,
        iface_use_tiling: bool = True,
    ):
        """
        初始化人脸识别器

        Args:
            recognition_model: InsightFace模型名称，默认'buffalo_l'
            gallery_path: 图库路径，包含按人员命名的子目录，每个子目录包含该人员的人脸图像
            threshold: 匹配阈值，默认0.4
            quality_threshold: 人脸质量阈值，默认0.7
            device: 计算设备，'auto'/'cpu'/'gpu'
            rebuild_gallery: 是否强制重建图库
            detection_model: （可选）YOLO 权重路径或名称。最佳实践：仅在你有“人脸专用 YOLO 权重”时启用。
            yolo_conf: YOLO检测置信度阈值，默认0.12
        """
        self.detection_model = detection_model
        self.recognition_model = recognition_model
        self.gallery_path = Path(gallery_path)
        self.threshold = threshold
        self.quality_threshold = quality_threshold
        self.det_size: Tuple[int, int] = (
            det_size,
            det_size,
        )  # InsightFace prepare det_size
        self.device = device
        self.rebuild_gallery = rebuild_gallery
        self.yolo_conf = float(yolo_conf)

        # InsightFace 检测策略：大图/小人脸场景启用平铺以提升召回
        # 注意：平铺会显著增加推理次数（尤其是 1080p/4K 视频）。
        self.iface_use_tiling = bool(iface_use_tiling)
        self.iface_tile_size = 960
        self.iface_tile_overlap = 0.25
        self.iface_full_max_side = 1600  # 超过则倾向直接平铺
        self.iface_tiling_min_faces = 1  # 全图检测不足则回退平铺

        # 人脸去重：平铺检测会产生重复框，使用 IoU-NMS 更稳
        self.face_nms_iou = 0.30

        # YOLO：仅作为可选候选框提案器（默认关闭）
        self.yolo_enabled = bool(self.detection_model)
        self.yolo_imgsz = 960
        self.yolo_use_tiling = True
        self.yolo_tile_size = 800
        self.yolo_tile_overlap = 0.25
        self.yolo_nms_thresh = 0.25

        # 模型实例
        self._detect_app: Optional[YOLOType] = None
        self._recogn_app: Optional[FaceAnalysis] = None

        # 图库（v2：每人多 embedding） + 匹配器
        self._gallery = Gallery(GalleryConfig(max_embeddings_per_person=5))
        self._matcher = CosineMatcher(
            MatcherConfig(
                threshold=float(self.threshold),
                margin=0.03,
                quality_alpha=0.10,
                min_threshold=0.30,
                max_threshold=0.65,
                unknown_label="未知",
            )
        )

        # 兼容旧字段：外部可能读取 gallery_embeddings / gallery_stats
        self.gallery_embeddings: Dict[str, np.ndarray] = {}
        self.gallery_stats: Dict[str, dict] = {}

        # 运行参数
        self.ctx_id = -1  # -1表示CPU，0表示第一个GPU

        # 内部自适应批处理：限制一次性送入 recognition.get_feat 的人脸数量，
        # 避免 CPU/内存带宽场景下“大 batch 反而更慢”。
        # 该值会在运行时根据耗时自适应调整；不暴露 CLI 参数。
        self._feat_chunk_size: Optional[int] = None
        self._feat_chunk_cap: int = 256

        # 初始化模型
        self._initialize_models()
        self._load_or_build_gallery()

    def _sync_gallery_views(self) -> None:
        """将新 Gallery 结构同步到旧的兼容字段。"""
        try:
            self.gallery_embeddings = dict(self._gallery.centroids)
        except Exception:
            self.gallery_embeddings = {}
        try:
            self.gallery_stats = dict(self._gallery.stats)
        except Exception:
            self.gallery_stats = {}

    def _initialize_models(self):
        """初始化InsightFace模型"""
        try:
            # 选择设备
            if self.device == "auto":
                try:
                    device = "gpu" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
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
            if self.yolo_enabled:
                det_key = str(self.detection_model)
                cached_yolo = _YOLO_CACHE.get(det_key)
                if cached_yolo is not None:
                    self._detect_app = cached_yolo
                else:
                    # 懒加载：避免在未启用 YOLO 时引入 ultralytics 的 import 开销
                    from ultralytics import YOLO

                    with suppress_fds():
                        self._detect_app = YOLO(self.detection_model)
                    _YOLO_CACHE[det_key] = self._detect_app
                logger.info(f"已加载 YOLO 模型(可选): {self.detection_model}")
            else:
                self._detect_app = None
                logger.info("YOLO 未启用（使用 InsightFace 检测主链路）")

            face_key = (
                str(self.recognition_model),
                tuple(str(p) for p in providers),
                int(self.ctx_id),
                tuple(int(x) for x in self.det_size),
            )
            cached_face = _FACEAPP_CACHE.get(face_key)
            if cached_face is not None:
                self._recogn_app = cached_face
            else:
                with suppress_fds():
                    self._recogn_app = FaceAnalysis(
                        name=self.recognition_model,
                        providers=providers,
                        allowed_modules=["detection", "recognition"],
                    )
                logger.info(f"已加载 InsightFace 模型: {self.recognition_model}")

                # 使用统一的 det_size 进行 prepare（会影响 detection 的输入尺度）
                buf = io.StringIO()
                with redirect_stdout(buf), redirect_stderr(buf):
                    self._recogn_app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)

                _FACEAPP_CACHE[face_key] = self._recogn_app
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise

    def _bbox_iou_xyxy(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算两个 xyxy bbox 的 IoU。"""
        ax1, ay1, ax2, ay2 = [float(x) for x in a]
        bx1, by1, bx2, by2 = [float(x) for x in b]
        xx1 = max(ax1, bx1)
        yy1 = max(ay1, by1)
        xx2 = min(ax2, bx2)
        yy2 = min(ay2, by2)
        w = max(0.0, xx2 - xx1)
        h = max(0.0, yy2 - yy1)
        inter = w * h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 1e-12 else 0.0

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
            image_files = [
                p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]

            logger.info(f"处理 {person_name}: {len(image_files)} 张图像")
            total_images += len(image_files)

            person_embeddings: List[np.ndarray] = []
            person_qualities: List[float] = []

            for img_file in image_files:
                try:
                    image = cv2.imread(str(img_file))
                    if image is None:
                        logger.warning(f"无法读取图像: {img_file}")
                        continue

                    # 图库构建最佳实践：优先整图 InsightFace 检测（必要时平铺回退）
                    faces = self._detect_faces_insightface_full(image)
                    if not faces and self.iface_use_tiling:
                        faces = self._detect_faces_insightface_tiled(image)
                    # 最后兜底：如果显式启用 YOLO，则尝试 YOLO -> crop -> InsightFace
                    if not faces and self.yolo_enabled:
                        yolo_boxes = (
                            self._detect_with_yolo_tiled(image)
                            if self.yolo_use_tiling
                            else self._detect_with_yolo(image)
                        )
                        faces = self._detect_faces_with_boxes(image, yolo_boxes)

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
                        embedding = l2_normalize(np.asarray(best_face.embedding, dtype=np.float32))
                        person_embeddings.append(embedding)
                        person_qualities.append(best_quality)

                        logger.info(f"  {img_file.name}: 质量{best_quality:.3f}, 尺寸{best_face.det_size}")
                        successful_images += 1
                    else:
                        logger.warning(f"  {img_file.name}: 人脸质量过低 ({best_quality:.3f}), faces={len(faces)}")

                except Exception as e:
                    logger.error(f"处理 {img_file} 失败: {e}")

            if person_embeddings:
                embs = np.stack(person_embeddings, axis=0)
                quals = np.asarray(person_qualities, dtype=np.float32)
                self._gallery.add_person_embeddings(person_name, embs, qualities=quals)
                logger.info(f"  ✅ 成功添加: {len(person_embeddings)} 个高质量embeddings")
            else:
                logger.warning(f"  ❌ {person_name} 没有合格的人脸图像")

        # 保存图库（v2）
        if self._gallery.person_to_embeddings:
            fp = self._gallery.save(
                self.gallery_path, threshold=self.threshold, quality_threshold=self.quality_threshold
            )
            self._sync_gallery_views()
            logger.info(
                f"图库构建完成: {len(self.gallery_embeddings)} 个人, {successful_images}/{total_images} 张图像 -> {fp}"
            )

    def _crop_and_resize(self, crop: np.ndarray, target: Tuple[int, int]) -> Tuple[np.ndarray, float, float]:
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

    def _detect_faces_insightface_full(self, image: np.ndarray) -> List:
        """使用 InsightFace 对整图检测。"""
        if self._recogn_app is None:
            return []
        try:
            faces = self._recogn_app.get(image) or []
        except Exception:
            return []
        for f in faces:
            try:
                f.det_size = tuple(self.det_size)
                f.enhancement = "insightface_full"
            except Exception:
                pass
        return faces

    def _detect_faces_insightface_full_batch(self, images: List[np.ndarray]) -> List[List]:
        """对一批图像执行 InsightFace 检测，并对所有检测到的人脸批量提取特征。

        目标：保持与逐张调用 `_recogn_app.get(img)` 一致的检测/对齐/embedding 语义，但减少
        recognition 模型的 session.run 次数（由每张脸一次 -> 每批一次）。
        """
        if self._recogn_app is None or not images:
            return [[] for _ in images]

        try:
            det_model = getattr(self._recogn_app, "det_model", None)
            rec_model = getattr(self._recogn_app, "models", {}).get("recognition")
        except Exception:
            det_model = None
            rec_model = None

        # 若无法获得底层模型，回退到逐张 get
        if det_model is None or rec_model is None or not hasattr(rec_model, "get_feat"):
            return [self._detect_faces_insightface_full(img) for img in images]

        from insightface.utils import face_align

        faces_batch: List[List] = [[] for _ in images]
        aligned: List[np.ndarray] = []
        face_ptrs: List[Tuple[int, int]] = []  # (img_idx, face_idx)

        for img_idx, img in enumerate(images):
            try:
                bboxes, kpss = det_model.detect(img, max_num=0, metric="default")
            except Exception:
                bboxes = np.zeros((0, 5), dtype=np.float32)
                kpss = None

            if bboxes is None or getattr(bboxes, "shape", (0,))[0] == 0:
                faces_batch[img_idx] = []
                continue

            cur_faces: List[Face] = []
            for i in range(int(bboxes.shape[0])):
                try:
                    bbox = bboxes[i, 0:4]
                    det_score = float(bboxes[i, 4])
                    kps = None
                    if kpss is not None:
                        kps = kpss[i]
                    face = Face(bbox=bbox, kps=kps, det_score=det_score)
                    cur_faces.append(face)
                except Exception:
                    continue

            faces_batch[img_idx] = cur_faces

            # 对齐并收集 batch
            for face_idx, f in enumerate(cur_faces):
                try:
                    if getattr(f, "kps", None) is None:
                        continue
                    aimg = face_align.norm_crop(
                        img,
                        landmark=f.kps,
                        image_size=int(getattr(rec_model, "input_size", (112, 112))[0]),
                    )
                    aligned.append(aimg)
                    face_ptrs.append((img_idx, face_idx))
                except Exception:
                    continue

        # 批量提特征（内部 micro-batch + 自适应 chunk size）
        if aligned:
            try:
                is_gpu = bool(getattr(self, "ctx_id", -1) >= 0)

                # 初始化 chunk size（按设备给一个稳健的起点）
                if self._feat_chunk_size is None:
                    self._feat_chunk_size = 256 if is_gpu else 32

                # 设备上限：GPU 允许更大；CPU 保守一些
                cap = int(self._feat_chunk_cap)
                if not is_gpu:
                    cap = min(cap, 64)

                chunk = int(max(1, min(int(self._feat_chunk_size), cap)))

                feats_out: List[np.ndarray] = []
                t_chunk_ema: Optional[float] = None

                # 分块执行 get_feat；同时根据 chunk 耗时自适应调整 chunk 大小
                offset = 0
                while offset < len(aligned):
                    sub = aligned[offset : offset + chunk]
                    t0 = time.time()
                    sub_feats = rec_model.get_feat(sub)
                    dt = max(1e-6, time.time() - t0)
                    feats_out.append(np.asarray(sub_feats))

                    # 更新 EMA（按 chunk 总耗时，而不是单脸耗时，避免噪声）
                    if t_chunk_ema is None:
                        t_chunk_ema = dt
                    else:
                        t_chunk_ema = 0.85 * float(t_chunk_ema) + 0.15 * float(dt)

                    # 自适应：chunk 太慢就缩小，太快就增大（但不超过 cap）
                    # CPU 的目标更保守，GPU 更激进。
                    slow_th = 0.60 if not is_gpu else 0.25
                    fast_th = 0.18 if not is_gpu else 0.10
                    if float(dt) > slow_th and chunk > 1:
                        chunk = max(1, chunk // 2)
                    elif float(dt) < fast_th and chunk < cap:
                        chunk = min(cap, max(chunk + 1, chunk * 2))

                    offset += len(sub)

                feats = np.concatenate(feats_out, axis=0) if len(feats_out) > 1 else feats_out[0]
                for j, (img_idx, face_idx) in enumerate(face_ptrs):
                    try:
                        faces_batch[img_idx][face_idx].embedding = np.asarray(feats[j]).flatten()
                    except Exception:
                        pass

                # 保存下一次调用的建议 chunk size（使得在同一机器上趋于稳定）
                self._feat_chunk_size = int(max(1, min(chunk, cap)))
            except Exception:
                # 回退：逐脸 get（保证正确性优先）
                for img_idx, img in enumerate(images):
                    for f in faces_batch[img_idx]:
                        try:
                            rec_model.get(img, f)
                        except Exception:
                            continue

        # 同步字段（与单图接口一致）
        for img_idx, faces in enumerate(faces_batch):
            for f in faces:
                try:
                    f.det_size = tuple(self.det_size)
                    f.enhancement = "insightface_full"
                except Exception:
                    pass

        return faces_batch

    def _detect_faces_insightface_tiled(self, image: np.ndarray) -> List:
        """InsightFace 平铺检测：对大图/密集小脸提升召回。"""
        if self._recogn_app is None:
            return []

        # 使用 detection + batch recognition（保持语义一致，但减少每脸一次 session.run）
        try:
            det_model = getattr(self._recogn_app, "det_model", None)
            rec_model = getattr(self._recogn_app, "models", {}).get("recognition")
        except Exception:
            det_model = None
            rec_model = None

        can_batch_rec = det_model is not None and rec_model is not None and hasattr(rec_model, "get_feat")
        if can_batch_rec:
            from insightface.utils import face_align

        h, w = image.shape[:2]
        tile = int(self.iface_tile_size)
        overlap = float(self.iface_tile_overlap)
        step = max(1, int(tile * (1.0 - overlap)))

        faces_all: List = []
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
                    if not can_batch_rec:
                        faces = self._recogn_app.get(crop) or []
                        for f in faces:
                            try:
                                bbox = f.bbox.astype(float)
                                bbox[0] += x1
                                bbox[2] += x1
                                bbox[1] += y1
                                bbox[3] += y1
                                f.bbox = bbox
                                if hasattr(f, "kps") and f.kps is not None:
                                    kps = np.asarray(f.kps).astype(float).reshape(-1, 2)
                                    kps[:, 0] += x1
                                    kps[:, 1] += y1
                                    f.kps = kps
                                f.det_size = tuple(self.det_size)
                                f.enhancement = "insightface_tiled"
                                faces_all.append(f)
                            except Exception:
                                continue
                        continue

                    # batch-rec path
                    bboxes, kpss = det_model.detect(crop, max_num=0, metric="default")
                    if bboxes is None or getattr(bboxes, "shape", (0,))[0] == 0:
                        continue

                    tile_faces: List[Face] = []
                    aligned: List[np.ndarray] = []
                    for i in range(int(bboxes.shape[0])):
                        try:
                            bbox = bboxes[i, 0:4]
                            det_score = float(bboxes[i, 4])
                            kps = kpss[i] if kpss is not None else None
                            face = Face(bbox=bbox, kps=kps, det_score=det_score)
                            tile_faces.append(face)
                            if kps is not None:
                                aimg = face_align.norm_crop(
                                    crop,
                                    landmark=kps,
                                    image_size=int(getattr(rec_model, "input_size", (112, 112))[0]),
                                )
                                aligned.append(aimg)
                            else:
                                aligned.append(None)
                        except Exception:
                            continue

                    if tile_faces:
                        # 提特征（仅对齐成功的）
                        try:
                            valid = [(idx, a) for idx, a in enumerate(aligned) if a is not None]
                            if valid:
                                feats = rec_model.get_feat([a for _, a in valid])
                                feats = np.asarray(feats)
                                for jj, (face_idx, _) in enumerate(valid):
                                    try:
                                        tile_faces[face_idx].embedding = feats[jj].flatten()
                                    except Exception:
                                        pass
                        except Exception:
                            # 回退：逐脸 get
                            for f in tile_faces:
                                try:
                                    rec_model.get(crop, f)
                                except Exception:
                                    continue

                        for f in tile_faces:
                            try:
                                bbox = np.asarray(f.bbox, dtype=float).reshape(-1)
                                bbox[0] += x1
                                bbox[2] += x1
                                bbox[1] += y1
                                bbox[3] += y1
                                f.bbox = bbox
                                if getattr(f, "kps", None) is not None:
                                    kps = np.asarray(f.kps).astype(float).reshape(-1, 2)
                                    kps[:, 0] += x1
                                    kps[:, 1] += y1
                                    f.kps = kps
                                f.det_size = tuple(self.det_size)
                                f.enhancement = "insightface_tiled"
                                faces_all.append(f)
                            except Exception:
                                continue
                except Exception:
                    continue

        # 平铺必然带来跨 tile 的重复框：用 IoU-NMS 做最终去重
        try:
            return self._dedupe_faces_nms(faces_all, image.shape, iou_thresh=self.face_nms_iou)
        except Exception:
            return faces_all

    def _dedupe_faces_nms(self, faces: List, image_shape: Tuple[int, int], iou_thresh: float = 0.30) -> List:
        """对人脸候选做 NMS 去重（解决重叠框/重复检测）。

        相比 `_merge_close_faces` 的中心点距离策略，IoU-NMS 更适合处理“框重叠但中心偏移略大”的情况，
        这在平铺检测跨 tile 回归时非常常见。
        """
        if not faces or len(faces) <= 1:
            return faces

        scored = []
        for f in faces:
            try:
                bbox = np.asarray(f.bbox, dtype=float).reshape(-1)
                if bbox.size != 4:
                    continue
            except Exception:
                continue

            # 评分融合：质量分为主，det_score 为辅
            try:
                q = float(self.assess_face_quality(f, image_shape))
            except Exception:
                q = 0.5
            try:
                det = float(getattr(f, "det_score", 0.8))
            except Exception:
                det = 0.8
            score = 0.7 * q + 0.3 * det
            scored.append((score, f, bbox))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        keep: List = []
        keep_boxes: List[np.ndarray] = []
        thr = float(iou_thresh)

        for score, f, bbox in scored:
            is_dup = False
            for kb in keep_boxes:
                if self._bbox_iou_xyxy(bbox, kb) >= thr:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(f)
                keep_boxes.append(bbox)

        return keep

    def _detect_with_yolo(self, image: np.ndarray, conf: float = None) -> List[Tuple[int, int, int, int]]:
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
            with suppress_fds():
                try:
                    results = self._detect_app(image, imgsz=self.yolo_imgsz)
                except TypeError:
                    # 兼容不同 ultralytics 版本的参数签名
                    results = self._detect_app(image)

            if not results:
                return boxes

            r = results[0]
            if r is None or not hasattr(r, "boxes") or r.boxes is None:
                return boxes

            # 关键优化：批量提取 xyxy/conf，避免逐 box 触发 .cpu().numpy() 同步
            xyxy_t = getattr(r.boxes, "xyxy", None)
            conf_t = getattr(r.boxes, "conf", None)
            if xyxy_t is not None and conf_t is not None:
                try:
                    if hasattr(xyxy_t, "detach"):
                        xyxy_t = xyxy_t.detach()
                    if hasattr(conf_t, "detach"):
                        conf_t = conf_t.detach()
                    xyxy_np = xyxy_t.cpu().numpy() if hasattr(xyxy_t, "cpu") else np.asarray(xyxy_t)
                    conf_np = conf_t.cpu().numpy() if hasattr(conf_t, "cpu") else np.asarray(conf_t)
                    conf_np = np.asarray(conf_np, dtype=np.float32).reshape(-1)
                    if xyxy_np is None or len(conf_np) <= 0:
                        return boxes
                    xyxy_np = np.asarray(xyxy_np)
                    if xyxy_np.ndim == 1 and xyxy_np.size >= 4:
                        xyxy_np = xyxy_np.reshape(1, -1)

                    n = min(int(xyxy_np.shape[0]), int(conf_np.shape[0]))
                    for i in range(n):
                        score = float(conf_np[i])
                        if score < conf:
                            continue
                        x1, y1, x2, y2 = xyxy_np[i][:4]
                        boxes.append((int(x1), int(y1), int(x2), int(y2)))
                    return boxes
                except Exception:
                    # 回退到逐 box 解析
                    pass

            # Fallback: 逐 box 兼容路径
            for box in r.boxes:
                xyxy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, "cpu") else np.array(box.xyxy)
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

    def _detect_with_yolo_batch(
        self, images: List[np.ndarray], conf: float = None
    ) -> List[List[Tuple[int, int, int, int]]]:
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
            with suppress_fds():
                results = self._detect_app(images)
            # Ultralytics YOLO 可能返回单个或列表，这里统一转换为列表
            if not isinstance(results, (list, tuple)):
                results = [results]

            for idx, (img, r) in enumerate(zip(images, results)):
                cur_boxes: List[Tuple[int, int, int, int]] = []
                try:
                    if r is None or not hasattr(r, "boxes") or r.boxes is None:
                        boxes_batch[idx] = []
                        continue

                    xyxy_t = getattr(r.boxes, "xyxy", None)
                    conf_t = getattr(r.boxes, "conf", None)
                    if xyxy_t is not None and conf_t is not None:
                        try:
                            if hasattr(xyxy_t, "detach"):
                                xyxy_t = xyxy_t.detach()
                            if hasattr(conf_t, "detach"):
                                conf_t = conf_t.detach()
                            xyxy_np = xyxy_t.cpu().numpy() if hasattr(xyxy_t, "cpu") else np.asarray(xyxy_t)
                            conf_np = conf_t.cpu().numpy() if hasattr(conf_t, "cpu") else np.asarray(conf_t)
                            conf_np = np.asarray(conf_np, dtype=np.float32).reshape(-1)
                            xyxy_np = np.asarray(xyxy_np)
                            if xyxy_np.ndim == 1 and xyxy_np.size >= 4:
                                xyxy_np = xyxy_np.reshape(1, -1)
                            n = min(int(xyxy_np.shape[0]), int(conf_np.shape[0]))
                            for i in range(n):
                                score = float(conf_np[i])
                                if score < conf:
                                    continue
                                x1, y1, x2, y2 = xyxy_np[i][:4]
                                cur_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                            boxes_batch[idx] = cur_boxes
                            continue
                        except Exception:
                            pass

                    for box in r.boxes:
                        xyxy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, "cpu") else np.array(box.xyxy)
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

    def _boxes_nms(
        self, boxes_with_scores: List[Tuple[int, int, int, int, float]], iou_thresh: float
    ) -> List[Tuple[int, int, int, int]]:
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
                    with suppress_fds():
                        try:
                            results = self._detect_app(crop, imgsz=self.yolo_imgsz)
                        except TypeError:
                            results = self._detect_app(crop)
                    if not results:
                        continue
                    r = results[0]
                    if r is None or not hasattr(r, "boxes") or r.boxes is None:
                        continue

                    xyxy_t = getattr(r.boxes, "xyxy", None)
                    conf_t = getattr(r.boxes, "conf", None)
                    if xyxy_t is not None and conf_t is not None:
                        try:
                            if hasattr(xyxy_t, "detach"):
                                xyxy_t = xyxy_t.detach()
                            if hasattr(conf_t, "detach"):
                                conf_t = conf_t.detach()
                            xyxy_np = xyxy_t.cpu().numpy() if hasattr(xyxy_t, "cpu") else np.asarray(xyxy_t)
                            conf_np = conf_t.cpu().numpy() if hasattr(conf_t, "cpu") else np.asarray(conf_t)
                            conf_np = np.asarray(conf_np, dtype=np.float32).reshape(-1)
                            xyxy_np = np.asarray(xyxy_np)
                            if xyxy_np.ndim == 1 and xyxy_np.size >= 4:
                                xyxy_np = xyxy_np.reshape(1, -1)
                            n = min(int(xyxy_np.shape[0]), int(conf_np.shape[0]))
                            for i in range(n):
                                score = float(conf_np[i])
                                if score < conf:
                                    continue
                                bx1, by1, bx2, by2 = xyxy_np[i][:4]
                                gx1 = x1 + int(bx1)
                                gy1 = y1 + int(by1)
                                gx2 = x1 + int(bx2)
                                gy2 = y1 + int(by2)
                                collected.append((gx1, gy1, gx2, gy2, score))
                            continue
                        except Exception:
                            pass

                    for box in r.boxes:
                        xyxy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, "cpu") else np.array(box.xyxy)
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
                        face.enhancement = "full_image"
                        all_faces.append(face)
                    except Exception:
                        continue
                logger.info(f"InsightFace 整图检测到 {len(all_faces)} 张人脸（回退路径）")
                return self._dedupe_faces_nms(all_faces, image.shape, iou_thresh=self.face_nms_iou)
            except Exception as e:
                logger.debug(f"InsightFace 整图回退失败: {e}")
                return []

        st = time.time()

        for bx1, by1, bx2, by2 in yolo_boxes:
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

                    if hasattr(face, "kps") and face.kps is not None:
                        kps = np.asarray(face.kps).astype(float).reshape(-1, 2)
                        kps[:, 0] = kps[:, 0] * sx + x1
                        kps[:, 1] = kps[:, 1] * sy + y1
                        face.kps = kps

                    face.det_size = target_det
                    face.enhancement = "unified_crop"

                    all_faces.append(face)
                except Exception:
                    continue

        ed = time.time()
        logger.info(f"人脸识别耗时: {ed - st:.2f} 秒，检测到 {len(all_faces)} 张人脸")

        try:
            return self._dedupe_faces_nms(all_faces, image.shape, iou_thresh=self.face_nms_iou)
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
        x1, y1, x2, y2 = result["bbox"]
        identity = result["identity"]
        similarity = result["similarity"]
        quality = result["quality"]

        # 根据身份和质量选择颜色
        if identity == "未知":
            color = (0, 0, 255)  # 红色
        elif quality >= 0.8:
            color = (0, 255, 0)  # 绿色（高质量）
        elif quality >= 0.6:
            color = (0, 255, 255)  # 黄色（中等质量）
        else:
            color = (0, 165, 255)  # 橙色（低质量）

        def _pick_text_color_for_bg(bg_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
            # Use perceived luminance to pick black/white for contrast.
            b, g, r = [float(x) for x in bg_bgr]
            # sRGB luma approximation (0..255)
            y = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return (0, 0, 0) if y >= 140.0 else (255, 255, 255)

        text_color = _pick_text_color_for_bg(color)

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 绘制关键点（如果有）
        if result["landmarks"] is not None:
            landmarks = result["landmarks"].astype(int)
            for x, y in landmarks:
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

        # 在同一帧上批量绘制文字，避免多次 PIL 转换（保持视觉效果一致）
        text_org = (bg_x1 + padding_x, bg_y1 + padding_y)
        q_w, q_h = measure_text_cn(quality_label, quality_font_size)
        q_org = (x1, min(image.shape[0] - q_h - 2, y1 + 15))
        try:
            draw_texts_cn(
                image,
                [
                    (label, text_org, int(label_font_size), text_color),
                    (quality_label, tuple(q_org), int(quality_font_size), text_color),
                ],
            )
        except Exception:
            # 兜底：保持旧行为
            draw_text_cn(image, label, text_org, font_size=label_font_size, color=text_color)
            draw_text_cn(image, quality_label, q_org, font_size=quality_font_size, color=text_color)

    def _load_or_build_gallery(self):
        """加载或构建图库"""
        if not self.rebuild_gallery:
            try:
                ok = self._gallery.load(self.gallery_path)
            except Exception:
                ok = False
            if ok:
                self._sync_gallery_views()
                logger.info(f"已加载图库: {len(self.gallery_embeddings)} 个人")
                return

        self._build_gallery()

    def _merge_close_faces(self, faces: List, image_shape: Tuple[int, int], center_tol: float = 0.012) -> List:
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
                best_score = getattr(faces[best_idx], "det_score", 0.5)

            for idx in group[1:]:
                try:
                    score = self.assess_face_quality(faces[idx], image_shape)
                except Exception:
                    score = getattr(faces[idx], "det_score", 0.5)
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
            det_score = getattr(face, "det_score", 0.8)
            quality_score += (det_score - 0.5) * 0.3

            # 3. 姿态评估（如果有关键点信息）
            if hasattr(face, "kps") and face.kps is not None:
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
        # 最佳实践：先用 InsightFace 整图检测；必要时再平铺提升小脸召回。
        h, w = image.shape[:2]
        max_side = max(h, w)

        faces: List = []
        if self.iface_use_tiling and max_side > self.iface_full_max_side:
            faces = self._detect_faces_insightface_tiled(image)
            if not faces:
                faces = self._detect_faces_insightface_full(image)
        else:
            faces = self._detect_faces_insightface_full(image)
            if (not faces or len(faces) <= self.iface_tiling_min_faces) and self.iface_use_tiling and max_side >= 900:
                tiled = self._detect_faces_insightface_tiled(image)
                if tiled:
                    faces = tiled

        # 可选：YOLO 提案作为“补充召回”，但只在显式启用时生效
        if self.yolo_enabled and self._detect_app is not None:
            try:
                yolo_boxes = (
                    self._detect_with_yolo_tiled(image) if self.yolo_use_tiling else self._detect_with_yolo(image)
                )
                yolo_faces = self._detect_faces_with_boxes(image, yolo_boxes)
                if yolo_faces:
                    faces = list(faces) + list(yolo_faces)
                    faces = self._dedupe_faces_nms(faces, image.shape, iou_thresh=self.face_nms_iou)
            except Exception:
                pass

        # 对主链路结果再做一次 IoU 去重，避免重叠框被当成多个人
        try:
            faces = self._dedupe_faces_nms(faces, image.shape, iou_thresh=self.face_nms_iou)
        except Exception:
            pass

        return faces

    def detect_faces_batch(self, images: List[np.ndarray]) -> List[List]:
        """对多张图像执行人脸检测与识别，尽可能在 YOLO 侧做 batch，以提高 GPU 利用率。

        Args:
            images: BGR 图像列表

        Returns:
            每张图像对应的人脸列表（与 detect_faces 单图接口兼容）
        """
        if not images:
            return []

        # 先批量跑整图 InsightFace（detection + batch recognition）
        full_batch = self._detect_faces_insightface_full_batch(images)

        results: List[List] = [[] for _ in images]

        # 可选：YOLO boxes 批量（仅当不启用平铺时，才能较好保持 batch 语义）
        yolo_boxes_batch: Optional[List[List[Tuple[int, int, int, int]]]] = None
        if self.yolo_enabled and self._detect_app is not None and not self.yolo_use_tiling:
            try:
                yolo_boxes_batch = self._detect_with_yolo_batch(images)
            except Exception:
                yolo_boxes_batch = None

        for idx, image in enumerate(images):
            h, w = image.shape[:2]
            max_side = max(h, w)

            # 与 detect_faces 单图逻辑保持一致
            faces = []
            if self.iface_use_tiling and max_side > self.iface_full_max_side:
                faces = self._detect_faces_insightface_tiled(image)
                if not faces:
                    faces = full_batch[idx] if idx < len(full_batch) else []
            else:
                faces = full_batch[idx] if idx < len(full_batch) else []
                if (
                    (not faces or len(faces) <= self.iface_tiling_min_faces)
                    and self.iface_use_tiling
                    and max_side >= 900
                ):
                    tiled = self._detect_faces_insightface_tiled(image)
                    if tiled:
                        faces = tiled

            # 可选：YOLO 提案补充召回（保持原逻辑）
            if self.yolo_enabled and self._detect_app is not None:
                try:
                    if yolo_boxes_batch is not None and idx < len(yolo_boxes_batch):
                        yolo_boxes = yolo_boxes_batch[idx]
                    else:
                        yolo_boxes = (
                            self._detect_with_yolo_tiled(image)
                            if self.yolo_use_tiling
                            else self._detect_with_yolo(image)
                        )
                    yolo_faces = self._detect_faces_with_boxes(image, yolo_boxes)
                    if yolo_faces:
                        faces = list(faces) + list(yolo_faces)
                        faces = self._dedupe_faces_nms(faces, image.shape, iou_thresh=self.face_nms_iou)
                except Exception:
                    pass

            try:
                faces = self._dedupe_faces_nms(faces, image.shape, iou_thresh=self.face_nms_iou)
            except Exception:
                pass

            results[idx] = list(faces) if faces else []

        return results

    def get_gallery_info(self) -> Dict:
        """获取图库信息"""
        info = {
            "total_persons": len(self.gallery_embeddings),
            "person_names": list(self.gallery_embeddings.keys()),
            "similarity_threshold": self.threshold,
            "quality_threshold": self.quality_threshold,
            "stats": self.gallery_stats,
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
                "bbox": (x1, y1, x2, y2),
                "identity": identity,
                "similarity": similarity,
                "quality": quality,
                "landmarks": getattr(face, "kps", None),
                "det_size": getattr(face, "det_size", None),
                "enhancement": getattr(face, "enhancement", "original"),
            }

            recognition_results.append(result)

            # 绘制结果
            self._draw(result_image, result, i)

            logger.info(f"检测到人脸 {i + 1}: {identity} (相似度: {similarity:.4f}, 质量: {quality:.3f})")

        # 保存结果
        if output_path:
            cv2.imwrite(output_path, result_image)
            logger.info(f"结果图像已保存至: {output_path}")

        return result_image, recognition_results

    def recognize_identity(
        self, embedding: np.ndarray, quality: float, debug: bool = False, topk: int = 5
    ) -> Tuple[str, float]:
        """
        身份识别，考虑质量因素

        Args:
            embedding: 人脸embedding
            quality: 人脸质量分数

        Returns:
            (身份, 相似度)
        """
        if not self._gallery.person_to_embeddings:
            return "未知", 0.0

        # 运行时允许外部修改 recognizer.threshold（例如 video_recognizer CLI 参数）
        self._matcher.config.threshold = float(self.threshold)

        emb = l2_normalize(np.asarray(embedding, dtype=np.float32))

        identity, sim, topk_list = self._matcher.match(
            emb,
            self._gallery.person_to_embeddings,
            quality=float(quality),
            topk_debug=int(topk),
        )

        if debug:
            logger.info(
                f"recognize_identity debug: quality={float(quality):.3f}, eff_thr={self._matcher.effective_threshold(float(quality)):.3f}, top{int(topk)}={topk_list}"
            )

        return identity, float(sim)
