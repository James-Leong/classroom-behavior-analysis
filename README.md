# 课堂行为分析系统

智能课堂行为分析系统，支持视频人脸识别、学生轨迹跟踪、行为识别与统计分析。

## 🎯 主要特性

- **两阶段处理流程**：人脸识别与行为识别完全解耦，提高灵活性
- **鲁棒轨迹跟踪**：Body bbox辅助跟踪，低头场景保持连续性
- **智能身份锁定**：多帧加强识别、滞回状态机、身份切换检测
- **零样本行为识别**：支持CLIP模型自定义行为类别
- **完整时间线记录**：从首次出现到最后消失，包含未识别阶段

## 🚀 快速开始

### 环境安装

本项目推荐使用 `uv` 管理 Python 环境（可选），也可以直接使用系统 Python + `pip`。

使用 `uv`（推荐）：

```bash
apt-get install pipx
pipx ensurepath
pipx install uv

# 安装项目依赖（默认）
uv sync

# 如果希望安装开发组依赖：
uv sync --group dev
```

如果不使用 `uv`，可直接安装依赖：

```bash
pip install -r requirements.txt
```

### 人脸识别

```bash
python video_recognizer.py data/video/20251115_1h.mp4 \
    --output-json outputs/face_results_1h.json \
    --gallery data/id_photo \
    --interval-frames 10 \
    --enable-person-detection
```

**输出**：包含人脸和body bbox的JSON文件

### 行为识别

```bash
python behavior_analyzer.py \
    --face-json outputs/face_results.json \
    --video data/video/20251115_clip.mp4 \
    --output-json outputs/behavior_stats.json \
    --model-type clip
```

**输出**：行为统计JSON文件

使用人体识别增强的行为识别：
```bash
python behavior_analyzer.py \
    --face-json outputs/face_results_1h.json \
    --video data/video/20251115_1h.mp4 \
    --person-detector models/yolo11n_classroom_context/weights/best.pt \
    --output-json outputs/behavior_finetuned.json
```

### 使用GPU服务器推理

首先将项目打包：
```bash
tar -zcvf behavior.tar.gz --exclude=".git" --exclude="./.ruff_cache" --exclude="./outputs" --exclude="./.venv" ./
```

然后将 `behavior.tar.gz` 上传到GPU服务器，解压后安装依赖并运行：

```bash
tar -zxvf behavior.tar.gz
uv sync  # 或 pip install -r requirements.txt
```

然后按照前述命令运行人脸识别和行为识别。

## 📚 文档

- [开发环境指南](./docs/DEV_ENVIRONMENT.md) - 根据不同的设备，配置开发环境的说明
- [CLIP使用指南](./docs/CLIP_USAGE.md) - 如何使用CLIP模型进行行为识别
- [YOLO身体框识别模型](./docs/YOLO_finetune_for_context.md) - 如何微调YOLO模型进行身体框识别

## 模型下载

### 图像检测模型
`yolo11n` 检测模型（`yolo11n.pt`）会在模块首次运行时自动下载到当前工作目录下。若下载速度过慢，请设置代理或手动下载：

```bash
export HTTP_PROXY="xxx"
export HTTPS_PROXY="xxx"
```

### 人脸识别模型
项目使用 `insightface` 的 `buffalo_l` 模型，会在首次运行时自动下载到 `~/.insightface/model`。若下载缓慢，可手动下载并解压到该目录：

```bash
# 使用镜像源下载
wget "https://gh-proxy.org/https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
mv buffalo_l.zip ~/.insightface/model/
cd ~/.insightface/model/
unzip buffalo_l.zip -d buffalo_l
rm buffalo_l.zip
```

## 中文字体异常

在部分 Ubuntu 系统中可能缺失中文字体，导致可视化图片中的中文标签乱码。可安装常用中文字体包以解决：

```bash
sudo apt update
sudo apt install -y fonts-noto-cjk fonts-wqy-zenhei fonts-wqy-microhei fonts-arphic-ukai fonts-arphic-uming
```

## 🏗️ 架构概览


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
