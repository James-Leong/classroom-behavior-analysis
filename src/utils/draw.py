from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence, Tuple

import warnings

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import FONT_LIST


_WARNED_NO_CJK_FONT = False


@lru_cache(maxsize=128)
def _load_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, font_size)


@lru_cache(maxsize=64)
def _get_best_font(font_size: int) -> ImageFont.ImageFont:
    """Return a font instance (cached) that best supports CJK on current OS."""
    for p in FONT_LIST:
        try:
            return _load_font(p, int(font_size))
        except Exception:
            continue
    return ImageFont.load_default()


def _warn_once_no_cjk_font_if_needed(texts: Iterable[str]) -> None:
    global _WARNED_NO_CJK_FONT
    if _WARNED_NO_CJK_FONT:
        return
    # If there is any non-ascii char, we likely need CJK-capable font.
    try:
        need_unicode = any(any(ord(ch) > 127 for ch in t) for t in texts)
    except Exception:
        need_unicode = False
    if not need_unicode:
        return
    # Probe whether any FONT_LIST entry is loadable at a common size.
    try:
        for p in FONT_LIST:
            try:
                _load_font(p, 16)
                return
            except Exception:
                continue
    except Exception:
        pass
    _WARNED_NO_CJK_FONT = True
    warnings.warn(
        "未找到可用的中文字体文件（FONT_LIST 全部加载失败），中文可能显示为方块/乱码。"
        "建议在 Linux 安装 fonts-noto-cjk 或 fonts-wqy-zenhei，"
        "或在 src/config.py 的 FONT_LIST 中加入可用字体路径。",
        RuntimeWarning,
    )


def draw_texts_cn(
    img: np.ndarray,
    items: Sequence[Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]],
) -> None:
    """Draw multiple unicode texts onto one frame with a single PIL conversion.

    Args:
        img: OpenCV BGR image, modified in-place.
        items: sequence of (text, (x, y), font_size_px, bgr_color)
    """
    if img is None or len(items) == 0:
        return

    try:
        _warn_once_no_cjk_font_if_needed([t for (t, _, _, _) in items])

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for text, org, font_size, bgr in items:
            try:
                font = _get_best_font(int(font_size))
            except Exception:
                font = ImageFont.load_default()

            # PIL uses RGB
            rgb_color = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            draw.text(tuple(org), str(text), font=font, fill=rgb_color)

        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback: draw one by one via OpenCV (unicode may still be broken)
        for text, org, font_size, bgr in items:
            font_scale = max(0.3, int(font_size) / 24.0)
            cv2.putText(
                img,
                str(text),
                tuple(org),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (int(bgr[0]), int(bgr[1]), int(bgr[2])),
                1,
                cv2.LINE_AA,
            )


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
    draw_texts_cn(img, [(str(text), tuple(org), int(font_size), (int(color[0]), int(color[1]), int(color[2])))])


def measure_text_cn(text: str, font_size: int = 14) -> Tuple[int, int]:
    """使用 PIL 测量文本像素尺寸，回退到 OpenCV 测量。

    性能：该函数在视频绘制中会被高频调用，因此增加 LRU 缓存。
    """

    return _measure_text_cn_cached(str(text), int(font_size))


@lru_cache(maxsize=4096)
def _measure_text_cn_cached(text: str, font_size: int) -> Tuple[int, int]:
    try:
        font = _get_best_font(int(font_size))

        # 使用 textbbox 得到精确尺寸
        dummy = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return int(width), int(height)
    except Exception:
        # 回退：OpenCV 近似测量
        font_scale = max(0.3, float(font_size) / 24.0)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        return int(w), int(h)
