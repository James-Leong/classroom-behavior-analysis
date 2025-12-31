# 输出数据

本目录用于前端离线演示加载分析结果（对应仓库根目录下 `outputs/` 的一份前端可读拷贝/切分版本）。推荐将“大 JSON”按时间分块放入 `chunks/`，再用 `face_manifest.json` 做索引。

## 目录结构

```text
fronts/public/data/outputs/
├── README.md
├── face_manifest.json
├── chunks/
│   ├── face_chunk_0.json
│   ├── face_chunk_1.json
│   └── ...
├── behavior_finetuned.json
└── debug_trace.json
```

## 文件说明

### face_manifest.json
- 作用：人脸/轨迹结果的“分块索引”，用于前端按播放时间按需加载对应 chunk。
- 内容：每个 chunk 的时间范围（或帧范围）与文件名（通常指向 `chunks/face_chunk_*.json`）。

### chunks/face_chunk_*.json
- 作用：分块后的阶段一输出（Face JSON v2 的子集），避免浏览器一次性加载/解析整段长视频结果。
- 内容：按帧组织的人脸检测、轨迹与身份相关字段（例如 bbox、identity、similarity、track_id、track_is_locked、body_bbox 等，具体字段取决于生成脚本版本）。
- 命名：`face_chunk_{i}.json`，`i` 从 0 开始递增。

### behavior_finetuned.json
- 作用：阶段二行为识别与统计结果（按学生聚合），供前端展示行为占比、分段时间线等。
- 内容：通常包含 timebase（fps、采样间隔等）、每位学生的观察时长、各行为的 segments（起止时间/帧、持续时长、占比）等。

### debug_trace.json
- 作用（可选）：行为识别调试轨迹，用于前端展示“裁剪框/采样帧/CLIP top-k/门控与平滑前后分数”等可解释证据。
- 说明：若不需要调试面板/证据链复核，可不提供该文件。
