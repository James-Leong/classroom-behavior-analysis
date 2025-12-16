"""独立的行为识别分析工具。

本工具基于人脸识别输出的JSON文件进行行为分析，实现了人脸识别与行为识别的完全解耦。

使用流程：
1. 首先运行 video_recognizer.py 生成包含人脸和body bbox的JSON文件
2. 然后运行本工具，读取JSON文件并进行行为识别分析
3. 输出独立的行为统计JSON文件

示例：
    # Step 1: 人脸识别
    python video_recognizer.py input.mp4 --output-json face_results.json

    # Step 2: 行为识别
    python behavior_analyzer.py --face-json face_results.json --video input.mp4 --output behavior_stats.json
"""

from __future__ import annotations

import argparse
import json
import time

from pathlib import Path
from typing import Dict

from src.behavior.pipeline import BehaviorPipelineConfig, run_behavior_pipeline_on_result
from src.behavior.stats import BehaviorSeriesConfig
from src.utils.log import get_logger

logger = get_logger(__name__)


def load_face_recognition_json(json_path: str) -> Dict:
    """加载人脸识别结果JSON文件。

    Args:
        json_path: JSON文件路径

    Returns:
        解析后的JSON数据

    Raises:
        FileNotFoundError: JSON文件不存在
        json.JSONDecodeError: JSON格式错误
    """
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"人脸识别JSON文件不存在: {json_path}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    schema_version = data.get("schema_version", "v1")
    logger.info(f"加载人脸识别结果: {json_path} (schema={schema_version})")

    # 验证必要字段
    required_fields = ["video", "fps", "frames", "mode"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"JSON文件缺少必要字段: {missing}")

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="视频行为识别分析工具：基于人脸识别JSON进行独立的行为分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法：分析所有已锁定身份的学生
  python behavior_analyzer.py --face-json results.json --video input.mp4
  
  # 指定特定学生进行分析
  python behavior_analyzer.py --face-json results.json --video input.mp4 \\
      --target 张三 --target 李四
  
  # 使用CLIP模型进行零样本行为识别
  python behavior_analyzer.py --face-json results.json --video input.mp4 \\
      --model-type clip --clip-model ViT-B/32
  
  # 自定义行为阈值
  python behavior_analyzer.py --face-json results.json --video input.mp4 \\
      --th-on 0.7 --th-off 0.5 --min-seconds 1.0
        """,
    )

    # 输入输出参数
    parser.add_argument(
        "--face-json",
        "-j",
        required=True,
        help="人脸识别结果JSON文件路径（由video_recognizer.py生成）",
    )
    parser.add_argument(
        "--video",
        "-v",
        required=True,
        help="原始视频文件路径（与人脸识别时使用的相同）",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        default="behavior_stats.json",
        help="输出行为统计JSON文件路径（默认: behavior_stats.json）",
    )

    # 分析目标
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="指定学生姓名进行分析（可重复多次），姓名需与图库目录名一致；不指定则分析所有已锁定身份",
    )
    parser.add_argument(
        "--ignore-lock-status",
        action="store_true",
        help="忽略锁定状态，对所有检测进行行为识别（包括未锁定的人脸）",
    )

    # 模型选择
    parser.add_argument(
        "--model-type",
        type=str,
        default="clip",
        choices=["kinetics", "clip"],
        help="行为识别模型类型: kinetics(Kinetics预训练) 或 clip(零样本自定义，推荐)",
    )

    # Kinetics模型参数
    parser.add_argument(
        "--kinetics-model",
        type=str,
        default="swin3d_t",
        choices=["swin3d_t", "mvit_v1_b", "s3d", "r3d_18"],
        help="Kinetics模型选择（仅当model-type=kinetics时有效）",
    )

    # CLIP模型参数
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="CLIP模型选择（仅当model-type=clip时有效）: ViT-B/32(快), ViT-B/16(平衡), ViT-L/14(准)",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=4,
        help="动作模型clip时间窗（秒，默认: 4）",
    )
    parser.add_argument(
        "--clip-frames",
        type=int,
        default=16,
        help="动作模型采样帧数（默认: 16）：在时间窗口内均匀采样此数量的帧",
    )

    # 设备参数
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda"],
        help="计算设备：auto/cpu/gpu（默认auto：有CUDA就用GPU）",
    )

    # 行为检测阈值
    parser.add_argument(
        "--th-on",
        type=float,
        default=0.60,
        help="行为进入阈值（默认: 0.60）",
    )
    parser.add_argument(
        "--th-off",
        type=float,
        default=0.45,
        help="行为退出阈值（默认: 0.45）",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=1,
        help="最短行为持续时间（秒，默认: 1）",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.30,
        help="同类行为段合并间隙（秒，默认: 0.30）",
    )

    args = parser.parse_args()

    # 加载人脸识别结果
    try:
        face_data = load_face_recognition_json(args.face_json)
    except Exception as e:
        logger.error(f"加载人脸识别JSON失败: {e}")
        return

    # 验证视频文件
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"视频文件不存在: {args.video}")
        return

    # 检查JSON中记录的视频路径是否匹配（仅警告，不阻止）
    json_video = face_data.get("video", "")
    if json_video and Path(json_video).name != video_path.name:
        logger.warning(f"JSON中记录的视频文件 ({json_video}) 与指定视频文件 ({args.video}) 不匹配，请确认是否正确")

    # 构建行为识别配置
    behavior_config = BehaviorPipelineConfig(
        enabled=True,
        target_names=args.target,
        model_type=str(args.model_type),
        action_model_name=str(args.kinetics_model),
        clip_model_name=str(args.clip_model),
        device=str(args.device),
        clip_seconds=float(args.clip_seconds),
        clip_num_frames=int(args.clip_frames),
        person_detector_weights=None,  # Not needed, using body_bbox from JSON
        person_conf=0.25,
        series_cfg=BehaviorSeriesConfig(
            th_on=float(args.th_on),
            th_off=float(args.th_off),
            min_duration_seconds=float(args.min_seconds),
            merge_gap_seconds=float(args.merge_gap),
        ),
        ignore_lock_status=bool(args.ignore_lock_status),
    )

    logger.info("=" * 60)
    logger.info("开始行为识别分析")
    logger.info(f"  人脸识别JSON: {args.face_json}")
    logger.info(f"  视频文件: {args.video}")
    logger.info(f"  模型类型: {args.model_type}")
    if args.model_type == "clip":
        logger.info(f"  CLIP模型: {args.clip_model}")
    else:
        logger.info(f"  Kinetics模型: {args.kinetics_model}")
    logger.info(f"  分析目标: {args.target if args.target else '所有已锁定身份'}")
    logger.info("=" * 60)

    # 执行行为识别
    start_time = time.time()
    try:
        behavior_stats = run_behavior_pipeline_on_result(
            input_video=str(args.video),
            result=face_data,
            cfg=behavior_config,
        )

        if behavior_stats is None:
            logger.warning("行为识别未返回结果")
            return

        # 保存结果
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(behavior_stats, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"行为识别完成！耗时: {elapsed:.2f}秒")
        logger.info(f"结果已保存: {output_path}")

        # 打印统计摘要
        if "students" in behavior_stats:
            num_students = len(behavior_stats["students"])
            logger.info(f"分析了 {num_students} 位学生的行为")
            for name, stats in behavior_stats["students"].items():
                total_time = stats.get("total_observed_seconds", 0)
                behaviors = stats.get("behaviors", {})
                logger.info(f"  - {name}: 观察时长 {total_time:.1f}秒, {len(behaviors)} 种行为")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"行为识别失败: {e}", exc_info=True)
        return


if __name__ == "__main__":
    main()
