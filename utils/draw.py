import warnings
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 常见系统字体候选（macOS/Windows/Linux），按需扩展
FONT_LIST = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/AppleGothic.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/STHeiti.ttf",
    "C:\\Windows\\Fonts\\msyh.ttc",
    "C:\\Windows\\Fonts\\msyh.ttf",
    "C:\\Windows\\Fonts\\simsun.ttc",
    "C:\\Windows\\Fonts\\arialuni.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def draw_text_cn(img: np.ndarray, text: str, org: Tuple[int, int], font_size: int = 14, color=(255, 255, 255)):
    """
    在 OpenCV 图像上绘制中文或其他 Unicode 文本，优先使用 PIL 回退到 cv2.putText
    """
    try:
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

        rgb_color = (color[2], color[1], color[0])
        draw.text(org, text, font=font, fill=rgb_color)
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        font_scale = max(0.3, font_size / 24.0)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (color[0], color[1], color[2]), 1, cv2.LINE_AA)


def measure_text_cn(text: str, font_size: int = 14) -> Tuple[int, int]:
    """返回文本像素宽高，回退到 OpenCV 估算"""
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

        dummy = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width, height
    except Exception:
        font_scale = max(0.3, font_size / 24.0)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        return w, h
