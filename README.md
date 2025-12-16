# 课堂行为分析系统 v2

智能课堂行为分析系统，支持视频人脸识别、学生轨迹跟踪、行为识别与统计分析。

## 🎯 主要特性

- **两阶段处理流程**：人脸识别与行为识别完全解耦，提高灵活性
- **鲁棒轨迹跟踪**：Body bbox辅助跟踪，低头场景保持连续性
- **智能身份锁定**：多帧加强识别、滞回状态机、身份切换检测
- **零样本行为识别**：支持CLIP模型自定义行为类别
- **完整时间线记录**：从首次出现到最后消失，包含未识别阶段

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用脚本一键运行

```bash
# 分析所有学生
./scripts/run_analysis.sh classroom_video.mp4

# 指定特定学生
./scripts/run_analysis.sh classroom_video.mp4 --target 张三 --target 李四
```

### 手动两阶段流程

#### 阶段一：人脸识别

```bash
python video_recognizer.py data/video/20251115_clip.mp4 \
    --output-video outputs/tracklet_20251115_clip.mp4 \
    --output-json outputs/face_results.json \
    --gallery data/id_photo \
    --interval-frames 10 \
    --enable-person-detection
```

**输出**：包含人脸和body bbox的JSON文件

#### 阶段二：行为识别

```bash
python behavior_analyzer.py \
    --face-json outputs/face_results.json \
    --video data/video/20251115_clip.mp4 \
    --output-json outputs/behavior_stats.json \
    --model-type clip
```

**输出**：行为统计JSON文件

## 📚 文档

- [v2重构详细说明](docs/REFACTOR_V2.md) - 架构设计、技术细节、使用指南
- [更新日志](CHANGELOG.md) - 版本变更记录

## 🏗️ 架构概览

### v2 重构核心改进

1. **解耦设计**
   - 人脸识别 → JSON → 行为识别
   - 可多次运行行为识别，无需重复人脸识别

2. **轨迹身份回溯**
   - 锁定后回溯到首次出现帧
   - 基于embedding距离检测身份切换

3. **Body bbox跟踪**
   - 低头场景保持轨迹连续性
   - `max_lost`从8帧→20帧

4. **预留扩展接口**
   - `HeadPoseEstimator`抽象类
   - 未来可集成MediaPipe等实现

## 📁 项目结构

```
classroom-behavior-analysis/
├── video_recognizer.py          # 人脸识别CLI
├── behavior_analyzer.py         # 行为识别CLI（新增）
├── src/
│   ├── face/                    # 人脸识别模块
│   ├── video/                   # 视频处理模块
│   │   ├── recognizer.py        # 视频人脸识别器
│   │   ├── tracker.py           # 轨迹跟踪器（增强）
│   │   └── head_pose.py         # 头部姿态接口（预留）
│   ├── behavior/                # 行为识别模块
│   │   ├── pipeline.py          # 行为识别流程（解耦）
│   │   └── person_detector.py   # Person检测器
│   └── utils/                   # 工具模块
├── scripts/
│   └── run_analysis.sh          # 一键分析脚本
└── docs/
    └── REFACTOR_V2.md           # v2详细文档
```

## 🔧 主要参数

### 人脸识别 (video_recognizer.py)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable-person-detection` | 启用person检测获取body bbox | `True` |
| `--max-lost` | 轨迹最大丢失帧数 | `20` |
| `--lock-threshold` | 身份锁定阈值 | `0.35` |
| `--switch-threshold` | 身份切换阈值 | `0.5` |

### 行为识别 (behavior_analyzer.py)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-type` | 模型类型：clip/kinetics | `clip` |
| `--clip-model` | CLIP模型 | `ViT-B/32` |
| `--target` | 指定分析学生（可多次） | `所有已锁定` |
| `--ignore-lock-status` | 分析所有检测（含未锁定） | `False` |

## 📊 JSON输出格式

### Schema v2

```json
{
  "schema_version": "v2",
  "video": "input.mp4",
  "fps": 30.0,
  "person_detection_config": {
    "enabled": true,
    "model": "yolo11n"
  },
  "frames": [
    {
      "frame": 0,
      "detections": [
        {
          "bbox": [100, 50, 150, 120],
          "body_bbox": [80, 50, 170, 200],
          "face_detection_status": "normal",
          "track_display_identity": "张三",
          "track_is_locked": true
        }
      ]
    }
  ],
  "tracklets": [
    {
      "id": 1,
      "first_detected_frame": 0,
      "lock_history": [...]
    }
  ]
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📝 许可

见 [LICENSE](LICENSE) 文件。