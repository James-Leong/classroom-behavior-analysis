from pathlib import Path

from src.face.recognizer import FaceRecognizer
from src.utils.log import get_logger

# 配置日志
logger = get_logger(__name__)


# 使用示例和测试函数
def main():
    """主函数：演示增强版识别器的使用"""

    # 创建增强版识别器
    recognizer = FaceRecognizer(
        gallery_path="data/id_photo",
        threshold=0.4,  # 降低阈值以提高召回率
        quality_threshold=0.6,  # 适中的质量要求
        det_size=320,
        device="auto",
        rebuild_gallery=True,  # 强制重建图库以确保最新数据
    )

    # 测试图像
    test_images = [
        "data/hangzhou.jpeg"  # 测试图像
    ]

    logger.info("=== 人脸识别测试 ===")

    for img_path in test_images:
        if not Path(img_path).exists():
            continue

        logger.info(f"🔍 测试: {img_path}")

        try:
            # 识别
            result_img, results = recognizer.proccess(img_path, f"output_{Path(img_path).stem}.jpg")

            if len(results) == 0:
                logger.warning("未检测到人脸")
            else:
                logger.info(f"检测到 {len(results)} 个人脸")

                for i, result in enumerate(results):
                    identity = result["identity"]
                    similarity = result["similarity"]
                    quality = result["quality"]

                    if identity == "未知":
                        logger.info(f"人脸 {i + 1}: {identity} (相似度: {similarity:.4f}, 质量: {quality:.3f})")
                    else:
                        logger.info(f"人脸 {i + 1}: {identity} (相似度: {similarity:.4f}, 质量: {quality:.3f}) ✅")

        except Exception as e:
            logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    main()
