# Windows 下 PyTorch DLL 加载问题修复指南

## 问题现象

在 Jupyter Notebook 中使用本地 LLM（Qwen3.5-9B + bitsandbytes 4-bit 量化）时报错：

```
[WinError 1114] 动态链接库(DLL)初始化例程失败。
Error loading "d:\Anaconda\envs\specllm\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
```

但直接用命令行运行相同的代码却正常。

## 根本原因

Windows 的 DLL 搜索机制是**全局共享**的。如果 `numpy`、`pandas` 等库先于 `torch` 被 import，它们会加载自己的 VC++ 运行时 DLL。当 `torch` 随后加载 `c10.dll` 时，发现所需的依赖已经被其他版本占用，导致初始化失败。

**关键：import 顺序决定成败。**

| 场景 | import 顺序 | 结果 |
|------|-------------|------|
| 命令行运行 | `torch` → `numpy`（torch 先加载） | 正常 |
| Jupyter cell-1 | `numpy` → `pandas` → ... → 后面才 `import torch` | DLL 冲突失败 |

## 修复方法

在 notebook 的**第一个 cell** 中，在所有其他 import 之前加上 `import torch`：

```python
import torch  # 必须在 numpy/pandas 之前，否则 Windows DLL 冲突

import os
import sys
import numpy as np
import pandas as pd
# ... 其他 import
```

## 验证步骤

1. 重启 Jupyter 内核（Kernel → Restart）
2. 从第一个 cell 开始重新运行
3. 确认输出中出现 `LLM: 使用本地模型（首次调用时加载）` 而非 DLL 错误

## 常见陷阱

- **修改 .py 文件后不生效**：Python 会使用 `__pycache__` 中的旧 .pyc 文件。修改后需要清除缓存：
  ```bash
  find . -name "*.pyc" -delete
  find . -name "__pycache__" -type d -exec rm -rf {} +
  ```
  或在 Jupyter 中重启内核。

- **`os.add_dll_directory()` 不能解决此问题**：在 `torch.load()` 或 `agent.load()` 中调用 `os.add_dll_directory(torch_lib_path)` 已经太晚了，因为 numpy 已经加载了冲突的 DLL。必须在 import 层面就保证顺序。

- **不要在 .py 文件顶部 `try: import torch`**：这只在该文件被首次 import 时有效。如果 notebook 已经先 import 了 numpy，再 import 这个 .py 文件，torch 仍然会冲突。修复必须在 notebook 的第一个 cell。

## 相关文件

- `cybench/llm/agent.py` — LLM agent，`load()` 中有 DLL 路径修复代码（作为辅助保障）
- `Adaptive_Features_Experiment.ipynb` — 实验 notebook，cell-1 必须先 import torch
