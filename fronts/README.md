# 课堂行为分析前端系统 (Classroom Behavior Analysis Frontend)

**在线演示地址**: [https://james-leong.github.io/classroom-behavior-analysis/](https://james-leong.github.io/classroom-behavior-analysis/)

这是一个用于分析和可视化课堂学生行为的综合仪表盘系统。作为课堂行为分析项目的可视化前端，它提供了实时视频回放、行为统计洞察以及详细的学生个人行为指标分析功能。

## 📋 功能特性

- **交互式仪表盘**:
  - **班级概览**: 可视化展示全班的行为统计数据（如听讲、读写、分心、使用电子设备等）。
  - **学生详情**: 支持下钻查看单个学生的具体表现指标和专注度分析。
  - **本地化支持**: 全面支持中文界面和行为标签显示。

- **高级视频播放器**:
  - 支持带有行为检测框（人脸及身体）的同步视频回放。
  - 交互式时间轴，允许快速定位和跳转到特定的行为事件发生时刻。

- **数据处理**:
  - **加载优化**: 针对大型分析结果数据集（如长视频分析结果）实现了分块加载策略，确保流畅的性能体验。
  - **离线/模拟支持**: 当后端真实数据不可用时，系统可优雅降级使用模拟数据进行演示。

- **调试工具**:
  - 实时推理概率可视化。
  - 支持 EMA (指数移动平均) 平滑处理后的评分跟踪。

## 🛠 技术栈

- **框架**: [React 18](https://reactjs.org/)
- **构建工具**: [Vite](https://vitejs.dev/)
- **语言**: [TypeScript](https://www.typescriptlang.org/)
- **状态管理**: [Zustand](https://github.com/pmndrs/zustand)
- **样式**: [Tailwind CSS](https://tailwindcss.com/)
- **可视化**: [Recharts](https://recharts.org/)
- **图标库**: [Lucide React](https://lucide.dev/)

## 🚀 快速开始

### 环境要求

- Node.js (v16 或更高版本)
- npm 或 yarn

### 安装步骤

1. 进入前端项目目录：
   ```bash
   cd fronts
   ```

2. 安装依赖：
   ```bash
   npm install
   ```

### 开发模式

启动本地开发服务器：
```bash
npm run dev
```
应用将在 `http://localhost:5173` 上运行。

### 生产构建

构建生产环境版本：
```bash
npm run build
```
构建产物将生成在 `dist` 目录下。

## 🌐 部署到 GitHub Pages（James-Leong.github.io）

本项目使用 Vite 构建，产物是纯静态文件（`dist/`），可以直接部署到 GitHub Pages。

### 方式 A：部署为主页（覆盖 James-Leong.github.io 根目录）

1. 在本仓库构建：
   ```bash
   cd fronts
   npm install
   npm run build
   ```
2. 克隆你的 GitHub Pages 仓库并覆盖根目录内容：
   ```bash
   git clone git@github.com:James-Leong/James-Leong.github.io.git
   rm -rf James-Leong.github.io/*
   cp -r dist/* James-Leong.github.io/
   ```
3. （可选）放入数据文件（否则页面会自动降级使用 mock 数据）：
   - 站点根目录需要有 `data/`，例如：
     - `data/video/20251115_1h.mp4`
     - `data/outputs/face_manifest.json`
     - `data/outputs/face_chunk_*.json`
     - `data/outputs/behavior_finetuned.json`
     - `data/outputs/debug_trace.json`（可选）
4. 在 `James-Leong.github.io` 仓库提交并推送：
   ```bash
   cd James-Leong.github.io
   git add -A
   git commit -m "Deploy classroom behavior analysis frontend"
   git push
   ```
5. 在 GitHub 仓库 Settings → Pages，选择从默认分支（通常是 `main`）的根目录部署。

### 方式 B：部署为子路径（保留你现有主页内容）

如果你不想覆盖根目录，可以把前端放在子目录（例如 `cba/`），访问地址会变成：
`https://james-leong.github.io/cba/`

```bash
git clone git@github.com:James-Leong/James-Leong.github.io.git
mkdir -p James-Leong.github.io/cba
rm -rf James-Leong.github.io/cba/*
cp -r dist/* James-Leong.github.io/cba/
```

如果需要真实数据，把 `data/` 放到 `James-Leong.github.io/cba/data/`。

## 📊 数据准备

为了在处理大型数据集（例如长达一小时的视频分析）时获得最佳性能，本项目采用了数据分块加载策略。

1. **生成数据**: 确保您的后端分析流水线已生成原始的 JSON 结果文件。
2. **处理数据**: 使用提供的 Python 脚本将大型 JSON 文件分割为易于管理的小块：
   ```bash
   # 请在项目根目录下运行
   python scripts/split_face_results.py
   ```
   该脚本将在输出目录中生成 `manifest.json` 和一系列数据块文件，这将显著减少首屏加载时间。

## 📂 项目结构

```
fronts/
├── src/
│   ├── components/     # 可复用的 UI 组件 (Timeline, VideoPlayer 等)
│   ├── constants/      # 共享常量配置 (行为标签, 颜色定义)
│   ├── pages/          # 主要应用页面 (Dashboard)
│   ├── store/          # 全局状态管理 (Zustand)
│   └── types/          # TypeScript 类型定义
├── public/             # 静态资源和数据文件
└── scripts/            # 辅助脚本 (如数据分割脚本)
```

## 📝 许可证

本项目是课堂行为分析系统的一部分。
