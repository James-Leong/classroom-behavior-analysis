YOLO 微调用于教室场景的身体框（包含设备/桌面）——知识文档

目标
- 提高行为识别中对“使用设备 / using_device”类的准确性。方法是微调目标检测器，使其输出的 `person` 框能够更频繁地包含桌面与设备（手机、笔记本），即学习生成“学生+工作区”的上下文框。

设计思路（Why）
- 问题：上游检测（YOLO）输出的 tight person 框常常不包含桌面与设备，导致基于 crop 的行为分类器（尤其是基于 CLIP 的零样本分类器）无法看到关键物体，误判为“reading”或“off_task”。
- 思路：保留预训练骨干（backbone），但通过小规模标注/微调（transfer learning）将输出 head 从 80 类替换为 1 类 `student_context`，并训练模型在教室视角下把人和桌面/设备都覆盖进框中。
- 替代或补充策略：固定或自适应扩大 person 框；检测 device（laptop/phone）并取 union；使用 HOI/两阶段模型检测人-物交互。微调优点是端到端学习上下文分布，鲁棒性更好。

数据准备（What & How）
1. 抽帧与预标注
   - 使用 `scripts/prepare_yolo_data.py` 从视频随机抽帧并用现有 `yolo11n.pt` 进行预标注，生成 `images/`、`labels/`（YOLO txt）和 `dataset.yaml`。
   - 该脚本也会生成：`label_studio_tasks.jsonl`（Label Studio 导入任务）、`labeling_config.xml`（标注配置）、`images.zip`（便于上传）。
   - 命令示例：
     ```bash
     python scripts/prepare_yolo_data.py --video data/video/20251115_clip.mp4 --num-images 50
     ```
2. 标注（重点）
   - 使用 Label Studio（推荐）或 LabelImg：`scripts/prepare_yolo_data.py` 会在 `data/yolo_finetune/` 下生成 `labeling_config.xml`（矩形框 + 单标签 `student_context`）供 Label Studio 使用。
   - 如需安装 Label Studio：`uv sync --group dev`（或自行 `pip install label-studio`）。
   - 标注原则：框必须包含“学生头、上身、双手，以及桌面上的设备/书本等交互对象”。不要把邻桌、背景设备包含进框。尽量紧凑但要包含交互对象。
   - 建议样本量：初始 50–200 张用于快速验证；若构建稳定模型建议 500+（带多样性：不同角度、不同摄像头、不同光照）。
3. 导出
   - 在 Label Studio 中导出 YOLO 格式（或 COCO 后转换为 YOLO）。导出后将 `images/` 与 `labels/` 放回 `data/yolo_finetune`（或新目录）。

YOLO 数据格式要点
- 图像：`images/frame_000123.jpg`
- 标签（每行）：`<class> <x_center> <y_center> <width> <height>`（均归一化）
- `dataset.yaml` 示例（脚本生成的 `path` 可能是绝对路径，也可使用相对路径）：
  ```yaml
  path: data/yolo_finetune
  train: images
  val: images
  nc: 1
  names: ["student_context"]
  ```

训练（训练脚本与参数）
- 提供脚本：`scripts/train_yolo_finetune.py`（基于 ultralytics YOLO API）
- 示例命令（使用 GPU）：
  ```bash
  python scripts/train_yolo_finetune.py --data data/yolo_finetune/dataset.yaml --epochs 50 --batch 16 --model yolo11n.pt
  ```
- 关键配置建议：
  - imgsz: 640（可根据显存与目标尺寸调整）
  - 冻结策略：数据少时先冻结骨干若干层（见 ultralytics 参数 freeze），只微调 head；若数据量较大可解冻全部训练。
  - 学习率：从 1e-3 开始实验，必要时较小（1e-4）以防过拟合。
  - 训练轮数：小数据集用 30–100 轮观察验证集表现。

如何用少量数据提高效果（实用技巧）
- 数据增强：水平翻转、随机尺度、色彩抖动、随机裁剪（注意不要把桌面切掉）。
- 类别均衡：尽量覆盖不同学生/桌面/设备场景。
- 使用更大的预训练模型（如果显存允许）进行微调可以提升泛化。

验证与评估
- 指标：mAP@0.5、Recall（特别关注是否覆盖设备）、IoU 分布（查看 IoU 是否能覆盖桌面）。
- 实验对照：在微调前后运行同一行为识别流水线（`behavior_analyzer.py` / pipeline）比较 `using_device` 精度/召回变化。

集成到现有流水线（部署）
1. 训练完成后会得到 `models/<project>/weights/best.pt`。
2. 把路径写入 `BehaviorPipelineConfig.person_detector_weights`（或直接替换 `yolo11n.pt` 的路径）：
   - 示例：在调用 `BehaviorPipelineConfig(... person_detector_weights='models/yolo11n_classroom_context/weights/best.pt')`
   - 或者直接在命令行中传入：`python behavior_analyzer.py ... --person-detector models/yolo11n_classroom_context/weights/best.pt`
3. 运行 `behavior_analyzer.py` 或 `pipeline.run` 测试整条链路效果。

推理细节（运行时）
- 我们的 `UltralyticsPersonDetector` 已支持传 `device`（cuda/cpu），`prepare_yolo_data.py` 和 `train_yolo_finetune.py` 都会优先用 CUDA（若可用）。
- Pipeline 中有三层优先逻辑：1) JSON 中的 `body_bbox` 2) person_detector 检测并匹配 face 3) face fallback expansion。微调后 person_detector 将更常命中，减少 fallback 使用率。

常见问题与解决
- Label Studio 的图片无法加载（file://）：要使用 Local Storage 并启动 Label Studio 时设置 `LOCAL_FILES_SERVING_ENABLED=true` 与 `LOCAL_FILES_DOCUMENT_ROOT`，并在 UI 中 Sync。
- ultralytics 版本差异：部分 API（device 参数、model.predict 参数）在不同版本差别较大；脚本中已做 TypeError 回退处理。
- `label-studio start --init` 报错：部分版本解析 CLI 存在 bug，推荐分两步 `label-studio init` 然后 `label-studio start ...`。
- GPU/显存不足：减小 `imgsz` 或 `batch`，或在训练时冻结更多层。

诊断建议（调试模型）
- 可视化检测结果：在若干验证图上画出微调前后检测框，对比是否包含桌面/设备。
- 统计覆盖率：计算包含设备区域的样本在微调前后的召回变化。
- 错误分析：对漏检与误检样本做分组（光照、角度、遮挡、多人桌面）以决定是否需要更多标注。

替代方案（如果微调效果不理想）
- 检测器 + 裁剪联合策略：检测 `person` 和 `laptop/phone`（新增 device 类）并取 union bbox。
- HOI 或关系检测模型：显式建模人体-物体交互，但实现复杂且需更多标注。

附：关键命令汇总
```bash
# 数据准备
python scripts/prepare_yolo_data.py --video data/video/20251115_clip.mp4 --num-images 100

# 初始化 Label Studio
label-studio init
# 启动 Label Studio 并配置本地文件服务
export LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT=$(pwd)/data/yolo_finetune/images
nohup label-studio start --no-browser --host 0.0.0.0 --port 8080 > labelstudio.log 2>&1 &
# 配置 Local Storage -> 指向 data/yolo_finetune/images -> Sync

# 训练微调
python scripts/train_yolo_finetune.py --data data/yolo_finetune/dataset.yaml --epochs 50 --batch 16

# 更新 pipeline 配置使用新模型
# 在 BehaviorPipelineConfig(person_detector_weights='models/yolo11n_classroom_context/weights/best.pt')
```

最后的话
- 微调 person 框以包含上下文是一个实际可行并且在 HOI 场景中常用的工程方法。先用少量数据快速迭代（50–200 张），观察 CLIP 行为分类器的性能提升，随后再决定是否扩大标注规模或采用更复杂的 HOI 方法。
- 需要的话我可以帮助：批量运行抽帧、检查 Label Studio 导入/导出、写 COCO→YOLO 转换脚本、或在训练后帮你把模型替入 pipeline 并跑一次对比实验。
## 训练文件打包与执行 (Packaging & Execution)

如果需要在其他机器或云端进行训练，请打包以下核心文件。

### 1. 打包命令
在项目根目录下运行：
```bash
tar -czvf yolo_finetune_package.tar.gz \
    data/yolo_finetune/dataset.yaml \
    data/yolo_finetune/images \
    data/yolo_finetune/labels \
    data/yolo_finetune/classes.txt \
    scripts/train_yolo_finetune.py \
    yolo11n.pt
```

### 2. 解压与训练命令
在目标机器上解压并运行：
```bash
# 解压
tar -xzvf yolo_finetune_package.tar.gz

# 安装依赖 (如果需要)
pip install ultralytics

# 运行训练 (推荐 100 epochs, batch 16)
python scripts/train_yolo_finetune.py \
    --data data/yolo_finetune/dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --name yolo11n_classroom_context
```
