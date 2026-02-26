"""可视化 Prefill 和 Decode 阶段的注意力图

用于生成类似论文中的 Prefill Attention 和 Decode Attention 可视化图
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import cv2
from typing import Optional, Dict, Tuple


def extract_prefill_attention(
    model,
    vision_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    提取 Prefill 阶段的注意力权重
    
    Args:
        model: PaliGemma 模型
        vision_tokens: 视觉 token embeddings [B, num_vision, D]
        text_tokens: 文本 token embeddings [B, num_text, D]
        attention_mask: 注意力 mask
    
    Returns:
        attention_weights: [B, num_heads, num_text, num_vision] - 文本对视觉的注意力
    """
    # 拼接视觉和文本 tokens
    inputs_embeds = torch.cat([vision_tokens, text_tokens], dim=1)  # [B, num_vision+num_text, D]
    
    # 获取语言模型
    llm = model.paligemma_with_expert.paligemma.language_model
    
    # 准备位置编码
    seq_len = inputs_embeds.shape[1]
    position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
    
    # 准备注意力 mask
    if attention_mask is None:
        attention_mask = torch.ones(
            (inputs_embeds.shape[0], seq_len),
            device=inputs_embeds.device,
            dtype=torch.bool
        )
    
    # 获取第一层的注意力（或最后一层）
    # 这里我们需要 hook 来提取注意力
    attention_weights = []
    
    def attention_hook(module, input, output):
        if hasattr(output, 'attentions') and output.attentions is not None:
            # 取第一层的注意力
            if len(output.attentions) > 0:
                attn = output.attentions[0]  # [B, num_heads, seq_len, seq_len]
                # 提取文本对视觉的注意力
                num_vision = vision_tokens.shape[1]
                text_to_vision_attn = attn[:, :, num_vision:, :num_vision]  # [B, num_heads, num_text, num_vision]
                attention_weights.append(text_to_vision_attn)
    
    # 注册 hook（这里需要根据实际模型结构调整）
    # 简化版本：直接计算交叉注意力
    return compute_cross_attention_manual(vision_tokens, text_tokens, llm)


def compute_cross_attention_manual(
    vision_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    llm,
) -> torch.Tensor:
    """
    手动计算文本对视觉的交叉注意力
    """
    # 获取第一层的注意力层
    if hasattr(llm, 'layers') and len(llm.layers) > 0:
        first_layer = llm.layers[0]
        if hasattr(first_layer, 'self_attn'):
            attn_layer = first_layer.self_attn
            
            # 拼接 tokens
            all_tokens = torch.cat([vision_tokens, text_tokens], dim=1)
            
            # 计算 Q, K, V
            hidden_states = first_layer.input_layernorm(all_tokens)
            query = attn_layer.q_proj(hidden_states)
            key = attn_layer.k_proj(hidden_states)
            value = attn_layer.v_proj(hidden_states)
            
            # Reshape for multi-head attention
            batch_size, seq_len, embed_dim = query.shape
            num_heads = attn_layer.num_heads
            head_dim = embed_dim // num_heads
            
            query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            # 计算注意力分数
            scale = 1.0 / (head_dim ** 0.5)
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            
            # Softmax
            attention_probs = F.softmax(attention_scores, dim=-1)
            
            # 提取文本对视觉的注意力
            num_vision = vision_tokens.shape[1]
            text_to_vision_attn = attention_probs[:, :, num_vision:, :num_vision]  # [B, num_heads, num_text, num_vision]
            
            # 平均所有 head
            text_to_vision_attn = text_to_vision_attn.mean(dim=1)  # [B, num_text, num_vision]
            
            # 平均所有文本 token
            text_to_vision_attn = text_to_vision_attn.mean(dim=1)  # [B, num_vision]
            
            return text_to_vision_attn
    
    # Fallback: 简单的余弦相似度
    vision_norm = F.normalize(vision_tokens, p=2, dim=-1)
    text_norm = F.normalize(text_tokens, p=2, dim=-1)
    text_avg = text_norm.mean(dim=1, keepdim=True)  # [B, 1, D]
    attention = (vision_norm * text_avg).sum(dim=-1)  # [B, num_vision]
    return attention


def extract_decode_attention(
    model,
    vision_tokens: torch.Tensor,
    past_key_values,
    current_token_embedding: torch.Tensor,
) -> torch.Tensor:
    """
    提取 Decode 阶段的注意力权重
    
    Args:
        model: PaliGemma 模型
        vision_tokens: 视觉 token embeddings [B, num_vision, D]
        past_key_values: KV cache
        current_token_embedding: 当前生成的 token embedding [B, 1, D]
    
    Returns:
        attention_weights: [B, num_vision] - 当前 token 对视觉的注意力
    """
    # 类似 Prefill，但只关注当前 token 对视觉的注意力
    llm = model.paligemma_with_expert.paligemma.language_model
    
    # 简化版本：使用余弦相似度
    vision_norm = F.normalize(vision_tokens, p=2, dim=-1)  # [B, num_vision, D]
    token_norm = F.normalize(current_token_embedding, p=2, dim=-1)  # [B, 1, D]
    
    # 计算注意力
    attention = (vision_norm * token_norm).sum(dim=-1)  # [B, num_vision]
    
    return attention


def reshape_attention_to_image(
    attention_weights: torch.Tensor,
    image_size: Tuple[int, int] = (224, 224),
    patch_size: int = 16,
    num_images: int = 1,
) -> np.ndarray:
    """
    将注意力权重重塑为图像空间
    
    Args:
        attention_weights: [B, num_vision] 或 [num_vision]
        image_size: 原始图像大小 (H, W)
        patch_size: patch 大小
        num_images: 图像数量
    
    Returns:
        attention_map: (H, W) 的注意力热力图
    """
    if attention_weights.ndim == 2:
        attention_weights = attention_weights[0]  # 取第一个 batch
    
    # 计算每个图像的 patch 数量
    h_patches = image_size[0] // patch_size
    w_patches = image_size[1] // patch_size
    patches_per_image = h_patches * w_patches
    
    # 处理多个图像的情况
    if num_images > 1:
        # 假设注意力权重是按图像顺序排列的
        # 这里简化处理：只可视化第一个图像
        attention_weights = attention_weights[:patches_per_image]
    
    # 确保长度匹配
    if len(attention_weights) > patches_per_image:
        attention_weights = attention_weights[:patches_per_image]
    elif len(attention_weights) < patches_per_image:
        # 填充
        padding = patches_per_image - len(attention_weights)
        attention_weights = torch.cat([
            attention_weights,
            torch.zeros(padding, device=attention_weights.device)
        ])
    
    # 转换为 numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    # 重塑为 patch 网格
    attention_map = attention_weights.reshape(h_patches, w_patches)
    
    # 上采样到原始图像大小
    attention_map = cv2.resize(
        attention_map,
        (image_size[1], image_size[0]),  # (W, H)
        interpolation=cv2.INTER_LINEAR
    )
    
    return attention_map


def visualize_prefill_decode_attention(
    image: np.ndarray,
    prefill_attention: torch.Tensor,
    decode_attention: torch.Tensor,
    output_path: str,
    patch_size: int = 16,
    num_images: int = 1,
    alpha: float = 0.6,
    cmap: str = 'jet',
):
    """
    可视化 Prefill 和 Decode 阶段的注意力图
    
    Args:
        image: 原始图像 (H, W, C)
        prefill_attention: Prefill 阶段的注意力 [B, num_vision] 或 [num_vision]
        decode_attention: Decode 阶段的注意力 [B, num_vision] 或 [num_vision]
        output_path: 输出路径
        patch_size: patch 大小
        num_images: 图像数量
        alpha: 叠加透明度
        cmap: 颜色映射
    """
    h, w = image.shape[:2]
    
    # 重塑注意力到图像空间
    prefill_map = reshape_attention_to_image(
        prefill_attention,
        image_size=(h, w),
        patch_size=patch_size,
        num_images=num_images,
    )
    
    decode_map = reshape_attention_to_image(
        decode_attention,
        image_size=(h, w),
        patch_size=patch_size,
        num_images=num_images,
    )
    
    # 归一化到 [0, 1]
    def normalize_map(attn_map):
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map
    
    prefill_map = normalize_map(prefill_map)
    decode_map = normalize_map(decode_map)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 第一行：Prefill Attention
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(prefill_map, cmap=cmap, interpolation='bilinear')
    axes[0, 1].set_title("(c) Prefill Attention", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 叠加 Prefill
    prefill_colored = plt.get_cmap(cmap)(prefill_map)[:, :, :3]
    prefill_overlay = (1 - alpha) * (image / 255.0) + alpha * prefill_colored
    axes[1, 0].imshow(prefill_overlay)
    axes[1, 0].set_title("Prefill Attention Overlay", fontsize=14)
    axes[1, 0].axis('off')
    
    # 第二行：Decode Attention
    im2 = axes[1, 1].imshow(decode_map, cmap=cmap, interpolation='bilinear')
    axes[1, 1].set_title("(d) Decode Attention", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 创建并排对比图（类似论文中的图）
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Prefill Attention 叠加
    prefill_colored = plt.get_cmap(cmap)(prefill_map)[:, :, :3]
    prefill_overlay = (1 - alpha) * (image / 255.0) + alpha * prefill_colored
    axes[0].imshow(prefill_overlay)
    axes[0].set_title("(c) Prefill Attention", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Decode Attention 叠加
    decode_colored = plt.get_cmap(cmap)(decode_map)[:, :, :3]
    decode_overlay = (1 - alpha) * (image / 255.0) + alpha * decode_colored
    axes[1].imshow(decode_overlay)
    axes[1].set_title("(d) Decode Attention", fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    side_by_side_path = output_path.replace('.png', '_side_by_side.png')
    plt.savefig(side_by_side_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存注意力可视化图: {output_path}")
    print(f"✅ 保存并排对比图: {side_by_side_path}")


def example_usage():
    """
    使用示例
    """
    # 这里需要根据实际模型和数据进行调整
    print("""
    使用示例：
    
    1. 在推理过程中提取注意力：
    
    # Prefill 阶段
    vision_tokens = model.embed_image(images)
    text_tokens = model.embed_language_tokens(text)
    prefill_attn = extract_prefill_attention(model, vision_tokens, text_tokens)
    
    # Decode 阶段（在生成过程中）
    current_token_emb = model.embed_language_tokens(current_token)
    decode_attn = extract_decode_attention(model, vision_tokens, past_kv, current_token_emb)
    
    # 可视化
    visualize_prefill_decode_attention(
        image=original_image,
        prefill_attention=prefill_attn,
        decode_attention=decode_attn,
        output_path="attention_visualization.png"
    )
    """)


if __name__ == "__main__":
    example_usage()





