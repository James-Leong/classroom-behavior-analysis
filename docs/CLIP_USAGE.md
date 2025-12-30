# CLIP Zero-Shot 行为识别使用指南

## 简介

本项目现在支持两种行为识别方法：

1. **Kinetics 预训练模型**（原有方案）- 使用 torchvision 预训练的视频动作模型
2. **CLIP Zero-Shot**（新方案）- 使用 OpenAI CLIP 实现自定义教室行为识别，无需训练

## 安装 CLIP

### ⚠️ 重要警告

**请勿使用以下命令**（会安装错误的包）：
```bash
# ❌ 错误 - 这会安装 PyPI 上的 clip 0.2.0（不是 OpenAI CLIP）
uv add clip
pip install clip
```

### 正确安装方法

```bash
# 方法1: 使用安装脚本（推荐）
bash scripts/install_clip.sh

# 方法2: 使用 uv（推荐）
uv pip install ftfy regex tqdm
uv pip install git+https://github.com/openai/CLIP.git

# 方法3: 使用 pip
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# 验证安装
python -c "import clip; print(clip.available_models())"
```

## 使用方法

### CLIP 模式（推荐用于教室场景）

```bash
# 基础用法 - 使用 ViT-B/32（最快）
python video_recognizer.py data/video/test.mp4 \
  --behavior \
  --behavior-model-type clip \
  --behavior-clip-model ViT-B/32 \
  --interval-frames 10 \
  --output-json outputs/clip_result.json

# 高精度模式 - 使用 ViT-B/16
python video_recognizer.py data/video/test.mp4 \
  --behavior \
  --behavior-model-type clip \
  --behavior-clip-model ViT-B/16 \
  --behavior-clip-subsample 2 \
  --interval-frames 10 \
  --output-json outputs/clip_high_acc.json

# 最高精度 - 使用 ViT-L/14（最慢但最准确）
python video_recognizer.py data/video/test.mp4 \
  --behavior \
  --behavior-model-type clip \
  --behavior-clip-model ViT-L/14 \
  --behavior-clip-subsample 1 \
  --interval-frames 30 \
  --output-json outputs/clip_best.json
```

### Kinetics 模式（向后兼容）

```bash
# 使用原有的 Kinetics 预训练模型
python video_recognizer.py data/video/test.mp4 \
  --behavior \
  --behavior-model-type kinetics \
  --behavior-model r3d_18 \
  --interval-frames 10 \
  --output-json outputs/kinetics_result.json
```

## 参数说明

### CLIP 专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--behavior-model-type` | `kinetics` | 模型类型：`clip` 或 `kinetics` |
| `--behavior-clip-model` | `ViT-B/32` | CLIP 模型：`ViT-B/32`、`ViT-B/16`、`ViT-L/14` |
| `--behavior-clip-subsample` | `4` | 每 N 帧处理一次（1=全部，4=每4帧） |

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--behavior` | - | 启用行为识别 |
| `--interval-frames` | `30` | 关键帧采样间隔 |
| `--batch-frames` | `1` | 批处理大小上限 |
| `--behavior-clip-seconds` | `2.0` | 动作片段时长（秒） |
| `--behavior-person-conf` | `0.25` | 人体检测置信度阈值 |

## 性能对比

### 速度对比（15秒视频，interval=10）

| 模型 | 推理时间 | 速度 | 精度 | 适用场景 |
|------|---------|------|------|---------|
| Kinetics r3d_18 | ~320s | ⭐⭐⭐⭐⭐ | ⭐⭐ | 快速原型 |
| Kinetics swin3d_t | ~610s | ⭐⭐⭐ | ⭐⭐⭐ | 通用场景 |
| CLIP ViT-B/32 (subsample=4) | ~400s | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **教室场景推荐** |
| CLIP ViT-B/16 (subsample=2) | ~600s | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高精度需求 |

### 识别质量对比

#### Kinetics 模型问题：
- ❌ 识别出不合理行为："folding napkins"（叠餐巾）、"crying"（哭泣90%）
- ❌ 类别粗糙：无法区分"认真写作业"和"涂鸦"
- ❌ 400个类别中仅15个适合教室（4%有效率）

#### CLIP 模型优势：
- ✅ 自定义教室行为：writing、reading、raising_hand、listening 等
- ✅ 语义理解：通过文本描述定义行为，更灵活
- ✅ 无不合理预测：类别完全可控

## 自定义行为类别

CLIP 默认识别以下12种教室行为：

```python
DEFAULT_BEHAVIORS = [
    "a student holding a pen and writing on a notebook or tablet",  # taking_notes
    "a student holding and reading a textbook or flipping through pages of a book",  # reading
    "a student sitting and looking straight ahead at the front of classroom",  # listening
    "a student holding a smartphone in hands and looking down at the phone screen with head lowered",  # using_phone
    "a student looking at a laptop computer screen on desk with hands on keyboard",  # using_phone
    "a student picking up a water bottle or cup, drinking water and putting it down",  # drinking
    "a student looking around in different directions, not looking straight ahead",  # distracted
    "a student with arms on desk and head lying down on the desk, or with eyes completely closed",  # sleeping
    "a student turning head to talk with a classmate sitting nearby",  # talking
]
```

### 修改行为类别

编辑 `src/behavior/action_model_clip.py`，修改 `DEFAULT_BEHAVIORS` 和 `DEFAULT_LABELS`：

```python
DEFAULT_BEHAVIORS = [
    "a student writing homework",
    "a student reading book",
    "a student raising hand",
    # 添加你的自定义行为...
]

DEFAULT_LABELS = [
    "writing",
    "reading", 
    "raising_hand",
    # 对应的短标签...
]
```

## 基于标注的优化迭代（推荐）

CLIP 在本项目里是 Zero-Shot（不训练），但可以通过“标注 → 评估 → 改提示词/策略 → 再评估”的方式持续提升效果。

### 1) 准备标注数据（manifest.jsonl）

推荐使用 JSONL（每行一个样本），路径：`data/annotations/manifest.jsonl`。字段建议与现有数据保持一致：

- `video_path`：原视频绝对路径
- `frame`：中心帧号（从 0 开始）
- `crop_bbox_xyxy`：用于行为识别的裁剪框（`[x1,y1,x2,y2]`）
- `annotator_label`：人工最终标签（细粒度/业务标签均可）
- `target_label`：当时的“候选/提案标签”（可选，但用于评估人工纠偏量很有用）
- `track_is_locked`、`student_id`、`quality`、`similarity`、`paths`：可选信息，便于筛选和回溯

### 2) 明确“细标签 → CLIP 粗标签”的映射

当前 CLIP 推理输出的是一组“粗标签”（见 `src/behavior/action_model_clip.py` 的 `DEFAULT_LABELS`）。如果你的人工标注是更细的业务标签（例如 `listening_upright/on_task_head_down/off_task`），需要做一个映射，常见做法是：

- `listening_upright` → `listening`
- `on_task_head_down` → `reading_or_writing`
- `off_task` → `distracted`
- `using_device` → `using_device`
- 其他不确定/遮挡/多人干扰 → `other`

### 3) 用标注评估并定位问题样本

使用 `scripts/calibrate_from_annotations.py` 生成评估报告（支持两类评估）：

- `heuristic`：评估 `target_label`（提案）与 `annotator_label`（人工）的差异，适合快速看“标注纠偏量/类别混淆”。
- `clip`：根据 `video_path + frame + crop_bbox_xyxy` 重新跑一次 CLIP（粗标签），再与人工映射后的粗标签比对，适合验证“当前提示词/温度/裁剪框”整体质量。

脚本默认读取 `data/annotations/manifest.jsonl` 并输出到 `outputs/calibration_report.json`。

### 4) 针对性优化（从高收益到低收益）

1. **改提示词（最常用）**：在 `src/behavior/action_model_clip.py` 的 `DEFAULT_PROMPTS_BY_LABEL` 里增加/替换多条描述，优先强化易混淆对：
   - `listening` vs `reading_or_writing`（抬头听课 vs 低头写读）
   - `distracted` vs `listening`（左顾右盼 vs 视线朝前）
   - `using_device` vs `reading_or_writing`（手机屏幕 vs 书本/作业本）
2. **调温度（temperature）**：`CLIPVideoActionModel(..., temperature=...)` 越大分布越“尖”，越小越“平”。在 `src/behavior/pipeline.py` 的 `BehaviorPipelineConfig.clip_temperature` 调整。
3. **启用/调不确定性门控**：在 `src/behavior/pipeline.py` 的 `BehaviorPipelineConfig` 中，通过 `uncertain_min_prob/uncertain_min_margin/uncertain_fallback_label` 把低置信度样本归入 `other`（或更保守的类别），减少误报。
4. **优化裁剪框（crop）**：如果 `crop_bbox_xyxy` 经常截掉手/桌面/手机等关键上下文，会导致 `using_device` 召回差。优先保证 crop 覆盖上半身与桌面区域（必要时改 `expand_bbox_xyxy` 的参数策略）。
5. **提升人体框质量（可选）**：如果 body bbox 不稳定，先提升 person 检测质量（例如使用微调过的 YOLO 权重）再跑 CLIP，往往比只改提示词更稳。

### 5) 固化策略并回归验证

每次改动建议只改一类因素（提示词/温度/门控/裁剪），然后复跑一次评估报告，对比：

- 混淆矩阵里主要混淆对是否下降
- `other` 是否合理上升（更保守）还是过度上升（过分不确定）
- 针对关键标签（如 `using_device`）的召回是否提升

## 性能优化建议

### GPU 利用率优化

1. **使用批处理**：
   ```bash
   --batch-frames 32
   ```

2. **调整子采样**：
   - `--behavior-clip-subsample 4`：每4帧处理一次（推荐）
   - `--behavior-clip-subsample 2`：更高精度，但慢2倍
   - `--behavior-clip-subsample 1`：处理所有帧（最慢）

3. **选择合适的模型**：
   - **快速原型**：`ViT-B/32` + subsample=4
   - **生产环境**：`ViT-B/32` + subsample=2
   - **最高精度**：`ViT-B/16` + subsample=2

### 降低处理时间

1. **增加采样间隔**：
   ```bash
   --interval-frames 30  # 从10增加到30，处理时间减少66%
   ```

2. **降低片段时长**：
   ```bash
   --behavior-clip-seconds 1.0  # 从2.0降低到1.0
   ```

3. **降低人体检测阈值**（提高覆盖率）：
   ```bash
   --behavior-person-conf 0.15
   ```

## 对比测试示例

```bash
# 测试1：CLIP ViT-B/32（快速模式）
python video_recognizer.py data/video/test.mp4 \
  --behavior --behavior-model-type clip \
  --behavior-clip-model ViT-B/32 \
  --behavior-clip-subsample 4 \
  --interval-frames 10 --max-seconds 15 \
  --output-json outputs/test_clip_fast.json

# 测试2：Kinetics r3d_18（对比）
python video_recognizer.py data/video/test.mp4 \
  --behavior --behavior-model-type kinetics \
  --behavior-model r3d_18 \
  --interval-frames 10 --max-seconds 15 \
  --output-json outputs/test_kinetics.json

# 比较结果
# 查看 behavior_stats.by_student 中的行为分布
```

## 故障排除

### 错误：AttributeError: module 'clip' has no attribute 'available_models'

**原因**：安装了错误的包（PyPI 上的 clip 0.2.0）

**解决方法**：
```bash
# 1. 卸载错误的包
uv pip uninstall clip
# 或
pip uninstall clip

# 2. 安装正确的 OpenAI CLIP
uv pip install git+https://github.com/openai/CLIP.git
# 或
pip install git+https://github.com/openai/CLIP.git

# 3. 验证
python -c "import clip; print(clip.available_models())"
```

### CLIP 未安装
```bash
# 使用 uv（推荐）
uv pip install ftfy regex tqdm
uv pip install git+https://github.com/openai/CLIP.git

# 或使用 pip
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### CUDA 内存不足
- 减小 batch-frames
- 增加 behavior-clip-subsample
- 使用更小的模型（ViT-B/32 而非 ViT-L/14）

### 识别精度不满意
- 增加 behavior-clip-subsample（减少到1或2）
- 使用更大的模型（ViT-B/16 或 ViT-L/14）
- 修改 DEFAULT_BEHAVIORS 使描述更精确

## 常见问题

**Q: CLIP 比 Kinetics 慢多少？**
A: ViT-B/32 + subsample=4 约慢 25%，但精度更适合教室场景。

**Q: 可以用中文描述行为吗？**
A: 当前版本需要英文。如需中文支持，可以使用 Chinese-CLIP（需额外集成）。

**Q: 如何回退到原来的方案？**
A: 使用 `--behavior-model-type kinetics` 即可，原有实现完全保留。

**Q: 输出格式是否改变？**
A: 完全兼容！输出的 `behavior_stats` 结构保持不变，只有行为标签不同。
