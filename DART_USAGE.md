# DART 使用指南

## 概述

DART (Differentiable Adaptive Region Tokenizer) 已成功集成到 pi0.5 中。DART 通过动态选择重要的图像 patch，实现自适应的 tokenization，可以在保持或提升性能的同时提高推理效率。

## 配置选项

在 `Pi0Config` 中添加了以下 DART 配置选项：

```python
# DART (Differentiable Adaptive Region Tokenizer) configuration
dart_enabled: bool = False  # 启用/禁用 DART
dart_num_patches: int = 196  # 目标 patch 数量（默认：196，即 14x14 for 224x224）
dart_scoring_backbone: str = "mobilenet_v3_small"  # 评分网络：可选 "mobilenet_v3_small", "mnasnet", "squeezenet", "efficientnet_b0"
dart_temperature: float = 1.0  # 可微分选择温度（越高越随机）
```

## 使用方法

### 1. 在配置中启用 DART

```python
from openpi.models import pi0_config

config = pi0_config.Pi0Config(
    dart_enabled=True,
    dart_num_patches=196,  # 对于 224x224 图像，196 = 14x14
    dart_scoring_backbone="mobilenet_v3_small",  # 轻量级，推荐
    dart_temperature=1.0,
    # ... 其他配置
)
```

### 2. 创建模型

```python
from openpi.models_pytorch import pi0_pytorch

model = pi0_pytorch.PI0Pytorch(config)
```

### 3. 训练和推理

DART 会自动在 patch embedding 阶段工作，无需额外代码。模型会根据图像内容动态选择重要的 patch。

## 评分网络选择

DART 支持多种轻量级 CNN 作为评分网络：

| Backbone | 参数量 | FLOPs | 准确率提升 | 推荐场景 |
|----------|--------|-------|-----------|----------|
| **mobilenet_v3_small** | 7M | 1.32G | +1.6% | ✅ **推荐**：平衡性能和效率 |
| **mnasnet** | 7M | 1.37G | +1.8% | 稍好性能，稍高计算 |
| **squeezenet** | 7M | 1.54G | +2.1% | 更好性能，更高计算 |
| **efficientnet_b0** | 10M | 2.41G | +2.9% | 最佳性能，最高计算 |

**建议**：对于大多数场景，使用 `mobilenet_v3_small` 即可获得良好的性能提升和效率平衡。

## 与其他 Token 优化技术的配合

DART 在 **patch embedding 阶段**工作，与其他 token 优化技术可以组合使用：

- **DART + ToMe**：DART 在 patch 级别选择，ToMe 在 transformer 层合并
- **DART + ToFu**：DART 在 patch 级别选择，ToFu 在 transformer 层过滤/融合
- **DART + V2Drop**：DART 在 patch 级别，V2Drop 在 LLM 层丢弃视觉 token

**注意**：同时启用多个技术可能会过度优化，建议先单独测试每个技术的效果。

## 性能预期

基于 DART 论文的实验结果：

- **准确率**：DeiT-Ti 提升 +1.6% (72.2% → 73.8%)
- **FLOPs**：轻微增加（约 +5%），但通过动态 patch 选择，实际推理速度可能提升
- **内存**：由于动态 patch 数量，内存使用可能减少

## 故障排除

### DART 未启用

如果看到警告信息：
```
[DART] ⚠️ DART enabled in config but module not available, using standard patch embedding
```

**解决方案**：
1. 确保 `torchvision` 已安装：`pip install torchvision`
2. 检查 `dart_pytorch.py` 文件是否存在
3. 检查导入路径是否正确

### 内存不足

如果遇到内存问题：
1. 减少 `dart_num_patches`（例如从 196 降到 144）
2. 使用更轻量的 `scoring_backbone`（例如 `mobilenet_v3_small`）
3. 降低 batch size

### 训练不稳定

如果训练时出现不稳定：
1. 降低 `dart_temperature`（例如从 1.0 降到 0.5）
2. 使用 warmup 策略
3. 降低学习率

## 示例代码

完整的使用示例：

```python
import torch
from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch

# 创建配置
config = pi0_config.Pi0Config(
    dart_enabled=True,
    dart_num_patches=196,
    dart_scoring_backbone="mobilenet_v3_small",
    dart_temperature=1.0,
    # ... 其他配置
)

# 创建模型
model = pi0_pytorch.PI0Pytorch(config)
model.eval()

# 准备输入（示例）
# observation = ...  # 你的观察数据

# 推理
# with torch.no_grad():
#     actions = model.sample_actions(device, observation)
```

## 技术细节

### DART 工作流程

1. **Patch 提取**：将输入图像分割成固定大小的 patch
2. **重要性评分**：使用轻量级 CNN（如 MobileNetV3）对每个 patch 进行评分
3. **动态选择**：使用可微分的 top-k 选择机制选择重要的 patch
4. **Patch Embedding**：只对选中的 patch 进行 embedding
5. **位置编码**：为选中的 patch 添加位置编码

### 可微分选择

DART 使用 Gumbel-Softmax 实现可微分的 top-k 选择，使得整个流程可以端到端训练。

## 参考

- DART 论文：https://arxiv.org/abs/2506.10390
- DART GitHub：https://github.com/HCPLab-SYSU/DART
- 集成评估报告：`DART_INTEGRATION_EVALUATION.md`



