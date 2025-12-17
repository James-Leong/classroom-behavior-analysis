# 开发环境指南

在最新的 NVIDIA GPU 上运行本项目时，只需要使用 `uv` 安装依赖即可，无需额外配置。
`uv` 会自动安装 [pyproject.toml](../pyproject.toml) 中指定的依赖和对应版本。

但是在某些老旧 GPU 或特定环境下，可能需要手动配置 CUDA 和相关库。以下是一些常见问题的解决方案。 

## Tesla P100

使用 Tesla P100 GPU 时，可能会遇到 CUDA 版本不兼容的问题。请按照以下步骤操作：
1. 修改 [pyproject.toml](../pyproject.toml) 文件的内容为：
```toml
[project]
name = "classroom-behavior-analysis"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
    "insightface>=0.7.3",
    "numpy<2.0.0",
    "onnx>=1.20.0",
    "onnxruntime>=1.23.2",
    "onnxruntime-gpu>=1.23.2",
    "onnxscript>=0.5.6",
    "openai-clip>=1.0.1",
    "opencv-python",
    "setuptools~=80.9.0",
    "torch==2.1.0",
    "torchaudio==2.1.0",
    "torchvision==0.16.0",
    "ultralytics>=8.3.235",
]

[dependency-groups]
dev = [
    "ipykernel",
    "pre-commit",
    "pytest",
    "ruff",
]

[tool.ruff]
line-length = 119

[[tool.uv.index]]
url = "http://mirrors.aliyun.com/pypi/simple/"
default = true

```

2. 运行以下命令重新安装依赖：
```bash
uv sync
```
