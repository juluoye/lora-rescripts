# V100 Compatibility Fix

Tesla V100 (SM 7.0) 与 cuDNN 9.11+ 的兼容性修复方案。

## 问题描述

cuDNN 9.11+ 移除了对 Compute Capability < 7.5 的支持，导致 V100 训练 Qwen Image 模型时崩溃：
```
RuntimeError: GET was unable to find an engine to execute this computation
```

## 解决方案

本目录提供两个独立的修复方案：

### 方案 1：运行时修复（推荐快速测试）
**文件：** `solution_c1_runtime_patch.py`

```python
from mikazuki.utils.v100_compat import solution_c1_runtime_patch
# 修复会自动应用
```

### 方案 2：权重转换（推荐生产环境）
**文件：** `solution_c2_weight_conversion.py`

```python
from mikazuki.utils.v100_compat.solution_c2_weight_conversion import convert_qwen_vae_for_v100

vae = AutoencoderKL.from_pretrained(...)
vae = convert_qwen_vae_for_v100(vae)
```

## 使用方法

详见项目根目录的 `docs/V100_COMPATIBILITY.md`

## 相关 Issue

https://github.com/WhitecrowAurora/lora-rescripts/issues/38
