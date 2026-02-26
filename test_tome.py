"""测试 ToMe 是否正常工作"""

import torch
import numpy as np
from openpi.training import config as _config
from openpi.models_pytorch import tome_pytorch

# 测试 ToMe 函数本身
print("=" * 60)
print("测试 1: ToMe 函数本身")
print("=" * 60)

# 创建测试 tokens
batch_size = 1
num_tokens = 256
dim = 1152
tokens = torch.randn(batch_size, num_tokens, dim)

print(f"原始 tokens shape: {tokens.shape}")

# 测试 ToMe
merged = tome_pytorch.apply_tome(tokens, ratio=0.5, enabled=True)
print(f"合并后 tokens shape: {merged.shape}")
print(f"Token 减少: {num_tokens} -> {merged.shape[1]} (减少了 {num_tokens - merged.shape[1]} 个)")

# 测试禁用状态
merged_disabled = tome_pytorch.apply_tome(tokens, ratio=0.5, enabled=False)
print(f"禁用时 tokens shape: {merged_disabled.shape}")
assert merged_disabled.shape == tokens.shape, "禁用时应该保持原样"

print("\n✅ ToMe 函数测试通过\n")

# 测试配置
print("=" * 60)
print("测试 2: 检查配置")
print("=" * 60)

config = _config.get_config("pi05_libero")
model_config = config.model

print(f"Config type: {type(model_config)}")
print(f"tome_enabled: {getattr(model_config, 'tome_enabled', 'NOT FOUND')}")
print(f"tome_ratio: {getattr(model_config, 'tome_ratio', 'NOT FOUND')}")
print(f"tome_metric: {getattr(model_config, 'tome_metric', 'NOT FOUND')}")

if hasattr(model_config, 'tome_enabled'):
    print(f"\n✅ ToMe 配置存在")
    if model_config.tome_enabled:
        print(f"✅ ToMe 已启用，ratio={model_config.tome_ratio}")
    else:
        print(f"⚠️  ToMe 已禁用")
else:
    print(f"\n❌ ToMe 配置不存在！")

print("\n" + "=" * 60)
print("测试 3: 检查模型创建")
print("=" * 60)

try:
    from openpi.models_pytorch import pi0_pytorch
    model = pi0_pytorch.PI0Pytorch(model_config)
    
    print(f"Model type: {type(model)}")
    print(f"Model has config: {hasattr(model, 'config')}")
    if hasattr(model, 'config'):
        print(f"Model config tome_enabled: {getattr(model.config, 'tome_enabled', 'NOT FOUND')}")
        print(f"Model config tome_ratio: {getattr(model.config, 'tome_ratio', 'NOT FOUND')}")
    
    print("\n✅ 模型创建成功")
except Exception as e:
    print(f"\n❌ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)







