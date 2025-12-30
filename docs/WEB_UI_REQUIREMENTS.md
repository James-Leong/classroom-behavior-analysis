# Web 前端展示页开发需求（课堂行为分析）

## 1. 背景与目标

当前仓库已具备一套“视频 → 人脸识别/轨迹/身体框 → 行为识别 → 统计输出”的离线流水线。下一阶段目标是提供一个 Web 前端页面，用于：

- 以可视化方式展示整套能力的运行流程与输出结果
- 以“行为识别”为典型例子，展示各阶段的关键中间产物与识别效果（可回放、可对齐到时间轴）
- （可选）调用大模型接口对课堂行为识别结果进行总结与讲评

## 2. 现有功能清单（补充版）

你已列出的功能：

1) 人脸识别  
2) 轨迹识别和身份合并  
3) 身体框识别  
4) 行为识别  
5) 其他微调优化  

建议补充并在前端可视化时覆盖的“能力点/运行阶段”：

- 两阶段解耦流程：`video_recognizer.py` 产出人脸/轨迹 JSON；`behavior_analyzer.py` 基于 JSON + 原视频做行为分析（解耦便于复用与迭代）
  - 入口：[video_recognizer.py](file:///mnt/l/project/classroom-behavior-analysis/video_recognizer.py)、[behavior_analyzer.py](file:///mnt/l/project/classroom-behavior-analysis/behavior_analyzer.py)
- 身份锁定/切换/解锁的滞回策略与锁定历史（用于降低抖动、减少误切换）
  - 相关字段：`track_display_identity`、`track_is_locked`、`lock_history`
- 低头/遮挡鲁棒性：body bbox 辅助跟踪（face missing 仍可 body-only 跟踪）
  - 相关字段：`body_bbox`、`face_detection_status=missing_body_tracked`
- 行为识别两类模型：Kinetics 预训练视频动作模型 vs CLIP Zero-Shot（可通过提示词定义标签）
  - 入口：[pipeline.py](file:///mnt/l/project/classroom-behavior-analysis/src/behavior/pipeline.py)、[action_model_clip.py](file:///mnt/l/project/classroom-behavior-analysis/src/behavior/action_model_clip.py)
- 行为后处理：时序平滑（EMA）、不确定性门控（top1 prob / margin）、分段（hysteresis）与统计（segments/ratio）
  - 配置入口：[BehaviorPipelineConfig](file:///mnt/l/project/classroom-behavior-analysis/src/behavior/pipeline.py)、[BehaviorSeriesConfig](file:///mnt/l/project/classroom-behavior-analysis/src/behavior/stats.py)
- 性能与工程优化：FFmpeg 硬件加速解码、批处理、动态 batch 调整、可选输出带标注视频

## 3. 输入/输出与可视化数据来源

### 3.0 本次演示数据约定（固定路径）

本期前端页面不需要执行任何 Python 推理脚本，只做“读取既有结果 + 媒体回放/切片/截图展示”。演示数据固定为：

- 原始视频：`data/video/20251115_1h.mp4`
- 人脸识别输出（大文件）：`outputs/face_results_1h.json`
- 行为识别输出（大文件）：`outputs/behavior_finetuned.json`
  - 说明：该输出使用了微调后的 YOLO person-detector 来提升 body bbox 质量（不是 CLIP 微调）

前端读取策略要求：

- 不强制一次性加载上述大 JSON 到内存；必须提供“索引/分块/按需加载”的方案（见第 6.3、10.2 节）
- 叠加展示使用“截图或视频切片”的方式呈现关键帧/关键片段，不要求逐帧实时渲染整段 1 小时视频

### 3.1 人脸识别输出（Face JSON，schema v2）

由 `video_recognizer.py` 生成。核心结构：

- 顶层 meta（视频、fps、抽帧间隔、person detector 配置等）：见 [recognizer.py:_build_header_meta](file:///mnt/l/project/classroom-behavior-analysis/src/video/recognizer.py#L86-L124)
- `frames[]`：每个导出帧的时间戳与 `detections[]`
  - `detections[]` 关键字段：`bbox`、`quality`、`similarity`、`identity`、`body_bbox`、`track_display_identity`、`track_is_locked`、`face_detection_status` 等（字段组装见 [recognizer.py:L1021-L1080](file:///mnt/l/project/classroom-behavior-analysis/src/video/recognizer.py#L1021-L1080)）
- `tracklets[]`：轨迹聚合信息（frames_count、resolved_identity、identities_freq、bbox_history、representative_bbox 等）：见 [serialize_tracklet](file:///mnt/l/project/classroom-behavior-analysis/src/utils/serializer.py#L96-L159)

前端可视化建议：

- 视频回放 + 叠加显示（人脸框、body 框、identity、similarity、quality、lock 状态、face_detection_status）
- “轨迹面板”：按 tracklet 展示时间范围、身份投票、锁定历史、代表帧跳转

### 3.2 行为识别输出（Behavior Stats JSON）

由 `behavior_analyzer.py` 生成。结构见 [build_behavior_stats](file:///mnt/l/project/classroom-behavior-analysis/src/behavior/report.py#L9-L90)：

- `timebase`：fps、used_frame_interval、sample_dt_seconds
- `denominator`：每位学生在镜头内可计入的 on_screen_seconds
- `by_student`：
  - `total_observed_seconds`
  - `behaviors[behavior].segments[]`：每个行为的时间段（start/end frame/time）、total_seconds、ratio

前端可视化建议：

- 班级总览：每位学生的观察时长、各行为占比（堆叠条形图/饼图）、关键事件列表（按段排序）
- 学生详情：行为时间线（Gantt/时间轴段）、段列表（可跳转视频时间）

### 3.3 行为识别“中间变量”展示要求（MVP 必须）

本期前端虽然不在浏览器里真实跑推理，但仍需要展示“功能执行过程”，尤其是行为识别的中间计算数据。仅靠 `behavior_stats.json` 无法满足，因为它不包含以下信息：

- 每个采样帧/每个学生 的原始 score 曲线（label → score_by_frame）
- 每次 clip 推理的裁剪框、采样帧索引、clip 截图/缩略图
- CLIP top-k（top1/top2 + margin）与不确定性门控是否触发
- 平滑前 vs 平滑后（EMA）对比

因此，本期演示数据除最终结果外，必须额外提供一份“Debug Bundle/Debug JSON”（离线预先生成并放到可被前端读取的位置），让前端能展示行为识别的中间变量。数据契约建议（输出任一即可）：

1) `outputs/run_<id>/debug_behavior.json`（推荐结构）
- `per_student_scores_raw`：student → label → {frame: score}
- `per_student_scores_ema`：student → label → {frame: score}
- `per_clip_inference[]`：每次推理的 (student, center_frame, frame_indices, crop_bbox, topk, margin, gated_flag)

2) 或 `outputs/run_<id>/debug_bundle.zip`
- `debug_behavior.json` + `thumbnails/*.jpg`（关键帧/裁剪图/clip 拼图）

前端的降级策略：

- 若 Debug 产物缺失，前端必须提示“中间变量不可用（缺少 debug bundle）”，但该场景不视为满足“过程展示”验收

### 3.4 离线“阶段产物清单”（manifest）建议（MVP 必须）

为保证离线模式也能按阶段展示过程与中间产物，建议提供一个可被前端读取的清单文件（manifest），用于描述“有哪些阶段、每个阶段对应哪些产物、产物在哪里”。推荐路径：

- `fronts/assets/indexes/run_manifest.json`

推荐结构（示例字段，允许增删，但必须能表达阶段与产物的对应关系）：

- `video_path`：如 `data/video/20251115_1h.mp4`
- `face_json_path`：如 `outputs/face_results_1h.json`
- `behavior_stats_path`：如 `outputs/behavior_finetuned.json`
- `debug_behavior_path`：如 `outputs/run_<id>/debug_behavior.json` 或解压后的等效路径
- `stages[]`：`[{stage_id, title, description, artifacts[]}]`
  - `artifacts[]`：可包含 `type`（frame/clip/json/table/log）、`path`、`time_range`（可选）、`student_id`（可选）

## 4. Web 产品形态与范围

### 4.1 必须支持的两种使用模式

1) 离线查看模式（MVP，本期范围）
- 输入：`data/video/20251115_1h.mp4`、`outputs/face_results_1h.json`、`outputs/behavior_finetuned.json`、行为 Debug Bundle（见第 3.3 节）、以及预先生成的截图/切片资源（见第 10.1 节）
- 推荐额外输入：`fronts/assets/indexes/run_manifest.json`（见第 3.4 节），用于驱动“按阶段展示过程”
- 前端只做可视化与检索，不负责跑推理/不调用 Python
- 允许只展示“关键片段与关键帧”，不要求全量逐帧可视化

2) 在线运行模式（增强，后续可选）
- 前端发起一次“分析任务”，后端执行现有 CLI/模块并产出结果文件
- 前端实时展示任务进度与阶段性产物

说明：第二种模式需要新增后端服务与任务系统（不建议把推理逻辑放到浏览器里）。

### 4.2 非目标（本期不做）

- 在浏览器端做模型推理（算力/依赖/体积不现实）
- 在线标注与标注回流闭环（可以后续扩展）
- 多租户权限系统（除非明确有需求）

## 5. 前端页面信息架构（IA）

### 5.1 页面列表

1) 任务主页 / 新建任务
- 输入：视频、图库（或选择已有图库）、参数（抽帧间隔、模型类型、clip 参数、阈值等）
- 输出：创建任务并跳转“任务详情”

2) 任务详情（Stage Timeline）
- 展示一个“阶段步骤条”（可折叠）：
  - 阶段 1：抽帧与解码
  - 阶段 2：人脸检测/识别
  - 阶段 3：轨迹跟踪/合并/锁定
  - 阶段 4：人体框（body bbox）
  - 阶段 5：行为识别（clip 裁剪、模型推理、平滑/门控）
  - 阶段 6：统计与导出
- 每个阶段可切换到该阶段的可视化视图（见第 6 节）

3) 结果总览（班级视角）
- 学生列表：姓名/观察时长/关键行为占比/异常提示（如 distracted 占比高、using_device 高）
- 行为分布图：按学生堆叠图、按时间聚合热力图
- 导出：下载 JSON、下载报告截图/CSV（可选）

4) 学生详情（个人视角）
- 视频播放器（支持跳转到具体 time）
- 行为时间线（segments）
- 段详情表（start/end time、时长、可一键跳转/截图）
- 叠加显示：face bbox、body bbox、identity、lock 状态

5) LLM 总结页/面板（可选）
- 以班级/单学生为粒度输出自然语言总结
- 展示引用的关键指标（可追溯，避免“凭空总结”）

## 6. 关键交互与可视化要求（以行为识别为例）

### 6.1 阶段可视化（MVP 可做的）

基于现有输出文件即可实现：

- 叠加框回放：
  - face bbox：`detections[].bbox`
  - body bbox：`detections[].body_bbox`（或无）
  - 显示身份：`track_display_identity`（或 fallback `identity`）
  - 锁定状态：`track_is_locked`
  - 质量与相似度：`quality`、`similarity`、`track_display_similarity`
- 轨迹详情：
  - `tracklets[]` 的 resolved_identity、identities_freq、representative_frames 跳转

行为部分（现有 behavior_stats）：

- 行为段时间线（segments）
- 行为段与视频联动跳转
- per_student 行为占比（ratio）

### 6.2 阶段可视化（中间变量，离线读取 Debug Bundle，MVP 必须）

前端通过读取第 3.3 节的 Debug Bundle/Debug JSON，实现“行为识别过程”的中间变量展示（无需在浏览器中跑推理）：

- 单个推理样本的“推理卡片”
  - center_frame、采样帧索引、crop_bbox、关键帧/裁剪图
  - top-k 与 margin、是否触发不确定性门控
  - 平滑前后对比（raw vs ema）
- 曲线视图
  - label score 随时间变化曲线（可叠加阈值线 th_on/th_off）
  - “门控触发点”标记

### 6.3 大文件与性能要求（MVP 必须满足）

- 视频为 1 小时大文件，前端默认只展示“关键帧截图/关键片段切片”，不要求逐帧渲染整段视频
- face_json / behavior_stats 可能很大，前端不得一次性全量解析并持久驻留内存
- 需要提供“按需加载”能力：
  - 以时间范围/帧范围为粒度的查询与渲染
  - UI 侧采用虚拟列表（segments/frames/tracklets）避免长列表卡顿
- 若仅靠前端无法满足性能，允许在离线准备阶段生成索引文件（例如 frame→byte_offset 的索引或按时间分片的 JSON）

## 7. LLM 总结能力（可选增强）

### 7.1 安全与架构要求

- 大模型 API Key 必须放在后端（环境变量/密钥管理），禁止在前端暴露
- 前端仅调用后端 `/api/summarize` 类接口
- 后端需做速率限制与输入大小限制（避免把整段逐帧数据直接塞进模型）

### 7.2 输入与输出建议

输入（推荐）：

- 以 `behavior_stats.json` 为主：
  - 班级：每位学生的 `total_observed_seconds` + 各行为 `total_seconds/ratio` + Top-N 行为段（按时长）
  - 单学生：行为段摘要（最长的几段、集中发生时间段）

输出（建议包含结构化字段，便于 UI 展示）：

- `summary_markdown`：可直接渲染
- `highlights[]`：要点列表（带 student_id、behavior、time_range）
- `warnings[]`：异常/高风险提示（如长期 distracted/using_device）

## 8. 后端接口与任务模型（供前端对接）

如果采用“在线运行模式”，建议提供最小可行 API（REST）：

- `POST /api/jobs`
  - 入参：视频上传/引用、gallery 路径、pipeline 参数
  - 出参：`job_id`
- `GET /api/jobs/{job_id}`
  - 出参：状态（queued/running/succeeded/failed）、当前阶段、进度、日志摘要
- `GET /api/jobs/{job_id}/artifacts`
  - 出参：可下载文件列表（video、face_json、behavior_stats、debug_bundle 等）
- `GET /api/jobs/{job_id}/frames/{t}`
  - 返回某时刻叠加渲染帧（PNG/JPEG）或返回 JSON 让前端自行绘制
- `POST /api/jobs/{job_id}/summarize`
  - 出参：LLM 总结结果

任务阶段枚举建议：

- `decode`
- `face_detect_recognize`
- `tracklet_update_merge_lock`
- `person_bbox`
- `behavior_infer`
- `behavior_postprocess`
- `export`

## 9. 验收标准（前端）

MVP（离线查看模式）验收：

- 能导入 `face_results.json` + 原视频，并在播放器上正确叠加 bbox 与身份信息
- 能导入 `behavior_stats.json`，展示班级总览与学生详情，且支持从任一 segment 跳转到视频时间
- UI 能清晰区分“未锁定/未知身份”“低质量/缺失但 body 跟踪”等状态
- 能按“阶段步骤条”展示过程，并能打开至少一个阶段的中间产物（推荐由 manifest 驱动）
- 能加载 Debug Bundle，并展示至少一种行为的 raw/ema 曲线与门控触发点，以及推理卡片（top-k、margin、crop、关键帧/裁剪图）

增强（在线运行模式）验收：

- 能发起一次任务并看到阶段进度与日志
- 能下载/打开产物，任务失败时能看到错误与定位建议

可选增强（LLM）验收：

- 能生成班级总结与单学生总结，且能点击回到对应证据片段（segments）

## 10. 技术实现建议（不强绑定）

- 前端建议选择：React/Vue 均可；重点在于“视频播放器 + canvas 叠加 + 时间轴组件 + 大数据列表性能”
- 可视化：时间轴（segments）建议使用虚拟列表 + 分段渲染，避免长视频卡顿
- 数据量：face_json 可能很大，前端应支持按需索引与分块加载（按 frame 范围加载或后端提供分页）

### 10.1 前端资源目录（fronts）约定（MVP 必须）

前端静态资源统一放在仓库根目录下 `fronts/`，用于存放演示所需的截图、切片视频、预计算索引等。约定结构：

- `fronts/index.html`、`fronts/styles.css`、`fronts/app.js`（或对应的打包产物目录）
- `fronts/assets/`
  - `frames/`：关键帧截图（jpg/png）
  - `clips/`：关键片段切片（mp4/webm）
  - `overlays/`：可选的预渲染叠加图（如 bbox 叠加后的帧）
  - `indexes/`：可选索引文件（见 10.2）

资源生成原则：

- 不在浏览器中执行 Python 推理
- 允许通过离线工具（Python/FFmpeg）提前生成截图/切片/索引，并由前端直接读取展示

### 10.2 大 JSON 的索引与分块加载建议（MVP 必须）

对于 `outputs/face_results_1h.json`、`outputs/behavior_finetuned.json`：

- 优先支持“分块格式”或“可索引格式”，避免一次性 JSON.parse 全量
- 允许的离线预处理方向（任选其一即可）：
  - 生成按时间分片的 JSON（例如每 1~5 分钟一个文件）并提供目录清单
  - 生成 JSONL/NDJSON（逐帧或逐段一行）以便流式读取
  - 生成索引文件（frame/time → byte_offset/文件名），前端按需 fetch range 或按文件加载
- 若后续引入后端服务，也可由后端提供分页/范围查询接口，前端只拉取当前视窗所需数据

### 10.3 Debug 数据生成工具（补充需求）

为满足第 3.3 节和第 6.2 节的“过程展示”需求，且避免前端跑重型推理，需要提供一个 Python 辅助工具（如 `scripts/extract_debug_trace.py`），用于生成前端所需的 `debug_trace.json`。

该工具职责：
- 输入：`face_results.json` + `video.mp4` + 指定时间段（如 `00:10:00-00:10:30`）
- 逻辑：复用 `behavior_analyzer.py` 的核心逻辑（crop, clip inference），但**仅跑指定片段**，并开启详细 Log 记录
- 输出：`debug_trace.json`，包含该片段逐帧的 scores、crops、top-k margins、gating status
- 目的：前端加载此文件后，可在播放该片段视频时，同步回放“拟真的”推理过程动画（曲线跳动、Log 滚动），实现“过程展示”效果。

### 10.4 模拟回放模式（Simulation Mode）交互细节
针对“演示/Review”场景，前端需提供一个沉浸式的“模拟运行”体验，核心在于**视觉上的动态变化**：

1. **入口与启动**：
   - 用户在“学生详情”或“轨迹片段”中，点击“查看推理过程（Debug）”按钮。
   - 系统加载对应的 `debug_trace.json`（如果文件不存在则提示不可用）。

2. **同步播放逻辑**：
   - **视频播放器**：正常播放原始视频。
   - **数据驱动动画**：前端监听视频的 `timeupdate` 事件，根据当前时间戳（`currentTime`）在 `debug_trace.json` 中查找最近的采样帧数据。

3. **侧边栏/浮层展示内容（随视频帧实时刷新）**：
   - **模型视野（Crop View）**：
     - 展示当前帧模型实际看到的“人脸/身体裁剪图”（从 debug 数据中获取 bbox 并在原图上动态裁剪，或直接显示 debug bundle 里的缩略图）。
     - *视觉效果*：随着视频播放，裁剪框内容会随人物移动而变化，体现“跟踪”过程。
   - **置信度动态柱状图（Score Bar）**：
     - 展示 `listening`, `reading`, `distracted` 等类别的实时得分。
     - *视觉效果*：柱状图高度随每一帧数据跳动（建议加 CSS `transition` 平滑过渡），直观展示模型的不确定性或判断倾向。
   - **平滑对比（Raw vs EMA）**：
     - 同时展示“原始预测（跳动剧烈）”与“EMA 平滑后（稳定）”的数值/曲线，体现算法对抖动的抑制作用。
   - **门控状态灯（Traffic Light）**：
     - **绿灯**：Pass（置信度高，margin 足够）。
     - **红灯**：Blocked（不确定性高，本次推理被丢弃）。
     - *视觉效果*：灯光颜色随每一帧的 `gating.gated` 状态切换，解释为什么某些时刻没有输出行为结果。

4. **交互控制**：
   - 拖动视频进度条时，右侧的 Debug 视图必须**立即同步**更新到对应时刻的状态，方便逐帧分析模型表现。

## 11. 开发验收核对单（Checklist）

在开发结束前，请对照以下核心目标进行自测：

- [ ] **结果展示完整性**：
  - 是否成功加载了 `outputs/face_results_1h.json` 和 `outputs/behavior_finetuned.json`？
  - 能否在班级总览中看到所有学生的统计数据？
  - 能否点击某个学生，跳转到其视频片段并看到正确的人脸/身体框叠加？

- [ ] **过程模拟真实性**：
  - 使用 `scripts/extract_debug_trace.py` 生成了一段测试数据的 Debug Trace。
  - 在前端播放该片段时，能否看到 Score 柱状图随视频跳动？
  - 门控状态灯是否会根据数据变红/变绿？
  - 裁剪图（Crop）是否跟随人物移动？

- [ ] **性能与体验**：
  - 打开 1 小时视频的大结果文件时，页面是否流畅（没有长时间卡死）？
  - 视频播放与叠加层是否对齐（没有明显的时间错位）？
