"""兼容入口：保留 `from video_recognizer import VideoFaceRecognizer`。

实现已拆分到 `src/video/` 包内；此文件仅保留薄封装与 CLI。
"""

from __future__ import annotations

import argparse
import time

from src.behavior.pipeline import BehaviorPipelineConfig
from src.behavior.stats import BehaviorSeriesConfig

from src.utils.log import get_logger
from src.video.recognizer import SCHEMA_VERSION, VideoFaceRecognizer

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="视频人脸识别：按间隔抽帧并识别")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("--output-video", "-o", help="输出带标注视频路径", default=None)
    parser.add_argument("--output-json", "-j", help="输出识别信息 JSON 路径", default="video_recognition_results.json")
    parser.add_argument("--gallery", "-g", help="图库路径", default="data/id_photo")
    parser.add_argument("--det-size", type=int, default=640, help="InsightFace det_size（默认 640）")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="计算设备：auto/cpu/gpu（默认 auto：有 CUDA 就用 GPU）",
    )
    parser.add_argument("--rebuild-gallery", action="store_true", help="强制重建图库 embedding")
    parser.add_argument(
        "--iface-use-tiling",
        type=int,
        default=1,
        choices=[0, 1],
        help="是否启用 InsightFace 平铺检测回退（默认 1）",
    )
    parser.add_argument("--interval", "-i", type=float, help="抽帧间隔（秒），默认 2.0", default=2.0)
    parser.add_argument("--interval-frames", "-f", type=int, help="抽帧间隔（帧数），如果指定则优先使用", default=None)
    parser.add_argument(
        "--mode", "-m", choices=["simple", "tracklet"], default="tracklet", help="识别模式：simple 或 tracklet"
    )
    parser.add_argument("--batch-frames", type=int, default=32, help="批量处理的帧数，>1 时启用多帧批识别")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="轨迹模式 IoU 阈值")
    parser.add_argument("--max-lost", type=int, default=8, help="轨迹模式允许的最大丢失帧数")
    parser.add_argument("--merge-sim-threshold", type=float, default=0.86, help="轨迹合并相似度阈值")
    parser.add_argument("--debug-identify", action="store_true", help="在识别时输出 top-k 相似度用于调试")
    parser.add_argument("--recognition-threshold", "-t", type=float, default=0.3, help="覆盖识别器的单帧相似度阈值")
    parser.add_argument("--tracklet-min-votes", type=int, default=2, help="轨迹内最小票数阈值")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧（用于快速调试/测试）")
    parser.add_argument("--max-seconds", type=float, default=None, help="最多处理多少秒（用于快速调试/测试）")
    parser.add_argument("--lock-threshold", type=float, default=0.35, help="轨迹锁定阈值（display_identity）")
    parser.add_argument("--lock-min-frames", type=int, default=2, help="锁定所需连续帧数")
    parser.add_argument("--switch-threshold", type=float, default=0.5, help="切换锁定身份阈值")
    parser.add_argument("--switch-min-frames", type=int, default=2, help="切换所需连续帧数")
    parser.add_argument("--unlock-threshold", type=float, default=0.35, help="解锁阈值（unknown 且低相似度时）")
    parser.add_argument("--unlock-grace-frames", type=int, default=3, help="解锁需要累计帧数（避免抖动）")
    parser.add_argument("--hold-unknown-frames", type=int, default=8, help="锁定后可容忍 unknown 的帧数")
    parser.add_argument(
        "--ffmpeg-codec",
        type=str,
        default="h264_nvenc",
        choices=["libx264", "h264_nvenc", "mpeg4"],
        help="指定 ffmpeg 编码器（默认自动优先 h264_nvenc，其次 libx264，最后 mpeg4）",
    )

    # Behavior analysis (video action model)
    parser.add_argument("--behavior", action="store_true", help="启用视频行为识别并输出 behavior_stats")
    parser.add_argument(
        "--behavior-target",
        action="append",
        default=None,
        help="指定学生姓名（可重复多次），姓名需与图库目录名一致；不指定则统计所有已锁定身份",
    )
    parser.add_argument(
        "--behavior-model",
        type=str,
        default="swin3d_t",
        choices=["swin3d_t", "mvit_v1_b", "s3d", "r3d_18"],
        help="视频动作模型（torchvision 预训练）",
    )
    parser.add_argument("--behavior-clip-seconds", type=float, default=2.0, help="动作模型 clip 时间窗（秒）")
    parser.add_argument("--behavior-clip-frames", type=int, default=16, help="动作模型采样帧数")
    parser.add_argument(
        "--behavior-person-weights",
        type=str,
        default="yolo11n.pt",
        help="person 检测权重路径（Ultralytics YOLO）",
    )
    parser.add_argument("--behavior-person-conf", type=float, default=0.25, help="person 检测置信度阈值")
    parser.add_argument("--behavior-th-on", type=float, default=0.60, help="行为进入阈值")
    parser.add_argument("--behavior-th-off", type=float, default=0.45, help="行为退出阈值")
    parser.add_argument("--behavior-min-seconds", type=float, default=0.50, help="最短行为持续时间（秒）")
    parser.add_argument("--behavior-merge-gap", type=float, default=0.30, help="同类行为段合并间隙（秒）")

    args = parser.parse_args()

    vfr = VideoFaceRecognizer(
        gallery_path=args.gallery,
        debug_identify=args.debug_identify,
        det_size=int(args.det_size),
        device=str(args.device),
        rebuild_gallery=bool(args.rebuild_gallery),
        iface_use_tiling=bool(int(args.iface_use_tiling)),
    )

    if args.recognition_threshold is not None:
        try:
            vfr.recognizer.threshold = float(args.recognition_threshold)
            logger.info(f"已设置单帧识别阈值: {vfr.recognizer.threshold}")
        except Exception:
            logger.warning("设置 recognition_threshold 失败，忽略")

    if args.mode == "tracklet":
        behavior_cfg = None
        if bool(args.behavior):
            behavior_cfg = BehaviorPipelineConfig(
                enabled=True,
                target_names=args.behavior_target,
                action_model_name=str(args.behavior_model),
                device=str(args.device),
                batch_size_cap=int(args.batch_frames),
                clip_seconds=float(args.behavior_clip_seconds),
                clip_num_frames=int(args.behavior_clip_frames),
                person_detector_weights=str(args.behavior_person_weights),
                person_conf=float(args.behavior_person_conf),
                series_cfg=BehaviorSeriesConfig(
                    th_on=float(args.behavior_th_on),
                    th_off=float(args.behavior_th_off),
                    min_duration_seconds=float(args.behavior_min_seconds),
                    merge_gap_seconds=float(args.behavior_merge_gap),
                ),
            )
        vfr.process_with_tracklets(
            args.input,
            args.output_video,
            args.output_json,
            frame_interval_sec=args.interval,
            frame_interval_frames=args.interval_frames,
            iou_threshold=args.iou_threshold,
            max_lost=args.max_lost,
            merge_similarity_threshold=args.merge_sim_threshold,
            tracklet_min_votes=args.tracklet_min_votes,
            batch_frames=args.batch_frames,
            max_frames=args.max_frames,
            max_seconds=args.max_seconds,
            lock_threshold=args.lock_threshold,
            lock_min_frames=args.lock_min_frames,
            switch_threshold=args.switch_threshold,
            switch_min_frames=args.switch_min_frames,
            unlock_threshold=args.unlock_threshold,
            unlock_grace_frames=args.unlock_grace_frames,
            hold_unknown_frames=args.hold_unknown_frames,
            ffmpeg_codec=args.ffmpeg_codec,
            behavior_config=behavior_cfg,
        )
    else:
        vfr.process(
            args.input,
            args.output_video,
            args.output_json,
            frame_interval_sec=args.interval,
            frame_interval_frames=args.interval_frames,
            batch_frames=args.batch_frames,
            ffmpeg_codec=args.ffmpeg_codec,
        )


if __name__ == "__main__":
    st = time.time()
    main()
    ed = time.time()
    logger.info(f"schema={SCHEMA_VERSION}, 总耗时: {ed - st:.2f} 秒")
