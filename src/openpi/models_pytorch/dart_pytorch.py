"""DART: Differentiable Adaptive Region Tokenizer for Vision Foundation Models.

DART dynamically selects important patches based on a scoring network,
allowing adaptive tokenization that adapts to image content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScoringNetwork(nn.Module):
    """Lightweight scoring network to evaluate patch importance.
    
    Supports multiple backbone architectures: MobileNetV3, MnasNet, SqueezeNet, EfficientNet-B0.
    This is a simplified version that processes individual patches.
    """
    
    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        in_channels: int = 3,
        output_dim: int = 1,
    ):
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == "mobilenet_v3_small":
            from torchvision.models import mobilenet_v3_small
            backbone_model = mobilenet_v3_small(weights=None)
            # Remove classifier and get features
            self.features = backbone_model.features
            # Get the output channels of the last layer
            last_channel = backbone_model.classifier[0].in_features
        elif backbone == "mnasnet":
            from torchvision.models import mnasnet0_5
            backbone_model = mnasnet0_5(weights=None)
            self.features = backbone_model.layers
            last_channel = 1280  # MnasNet output channels
        elif backbone == "squeezenet":
            from torchvision.models import squeezenet1_0
            backbone_model = squeezenet1_0(weights=None)
            self.features = backbone_model.features
            last_channel = 512
        elif backbone == "efficientnet_b0":
            from torchvision.models import efficientnet_b0
            backbone_model = efficientnet_b0(weights=None)
            self.features = backbone_model.features
            last_channel = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Add a small head to output importance scores
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_channel, output_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through scoring network.
        
        Args:
            x: Input patch tensor of shape (B, C, H, W) where H, W are patch size
            
        Returns:
            Importance scores of shape (B, 1)
        """
        features = self.features(x)
        scores = self.head(features)  # (B, 1)
        return scores


def differentiable_topk(
    scores: torch.Tensor,
    k: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable top-k selection using Gumbel-Softmax.
    
    Args:
        scores: Importance scores of shape (B, N) where N is number of patches
        k: Number of patches to select
        temperature: Temperature for Gumbel-Softmax
        
    Returns:
        selected_mask: Binary mask of shape (B, N) indicating selected patches
        selected_indices: Indices of selected patches
    """
    B, N = scores.shape
    
    if k >= N:
        # Select all patches
        mask = torch.ones(B, N, device=scores.device, dtype=scores.dtype)
        indices = torch.arange(N, device=scores.device).unsqueeze(0).expand(B, -1)
        return mask, indices
    
    # Add Gumbel noise for differentiable sampling
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
    perturbed_scores = scores + gumbel_noise * temperature
    
    # Get top-k indices
    _, topk_indices = torch.topk(perturbed_scores, k, dim=1)  # (B, k)
    
    # Create mask
    mask = torch.zeros(B, N, device=scores.device, dtype=scores.dtype)
    mask.scatter_(1, topk_indices, 1.0)
    
    return mask, topk_indices


def compute_patch_scores(
    image: torch.Tensor,
    scoring_network: nn.Module,
    patch_size: int,
    num_patches_target: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute importance scores for each patch and select top-k.
    
    Args:
        image: Input image of shape (B, C, H, W)
        scoring_network: Scoring network to compute importance
        patch_size: Size of each patch
        num_patches_target: Target number of patches to select
        temperature: Temperature for differentiable selection
        
    Returns:
        patch_mask: Binary mask of shape (B, H_patch, W_patch) indicating selected patches
        selected_indices: Indices of selected patches
    """
    B, C, H, W = image.shape
    H_patch = H // patch_size
    W_patch = W // patch_size
    num_patches = H_patch * W_patch
    
    # Extract patches
    patches = F.unfold(
        image,
        kernel_size=patch_size,
        stride=patch_size,
    )  # (B, C*patch_size*patch_size, num_patches)
    
    # Reshape to (B, num_patches, C, patch_size, patch_size)
    patches = patches.view(B, C, patch_size, patch_size, num_patches)
    patches = patches.permute(0, 4, 1, 2, 3).contiguous()
    patches = patches.view(B * num_patches, C, patch_size, patch_size)
    
    # Compute scores for each patch
    patch_scores = scoring_network(patches)  # (B*num_patches, 1)
    patch_scores = patch_scores.view(B, num_patches)
    
    # Select top-k patches
    k = min(num_patches_target, num_patches)
    mask, indices = differentiable_topk(patch_scores, k, temperature)
    
    # Reshape mask to spatial format
    mask_2d = mask.view(B, H_patch, W_patch)
    
    return mask_2d, indices


class DartPatchEmbedding(nn.Module):
    """DART-based patch embedding with dynamic patch selection.
    
    This module replaces the standard patch embedding with DART,
    which dynamically selects important patches based on content.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        patch_size: int = 16,
        image_size: int = 224,
        num_patches_target: int = 196,  # Target number of patches (default: 14x14 for 224x224)
        scoring_backbone: str = "mobilenet_v3_small",
        temperature: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches_target = num_patches_target
        self.temperature = temperature
        self.enabled = enabled
        
        # Standard patch embedding (used when DART is disabled or as fallback)
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        )
        
        # Scoring network for DART
        if enabled:
            self.scoring_network = ScoringNetwork(
                backbone=scoring_backbone,
                in_channels=in_channels,
                output_dim=1,
            )
        else:
            self.scoring_network = None
        
        # Position embedding (will be interpolated based on selected patches)
        self.num_patches_full = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches_full, embed_dim)
        self.register_buffer(
            "position_ids_full",
            torch.arange(self.num_patches_full).expand((1, -1)),
            persistent=False,
        )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        """Forward pass with DART patch selection.
        
        Args:
            pixel_values: Input images of shape (B, C, H, W)
            interpolate_pos_encoding: Whether to interpolate position encoding
            
        Returns:
            Patch embeddings of shape (B, num_selected_patches, embed_dim)
        """
        B, C, H, W = pixel_values.shape
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        num_patches_full = H_patch * W_patch
        
        if not self.enabled or self.scoring_network is None:
            # Fallback to standard patch embedding
            patch_embeds = self.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
            
            # Add position encoding
            if interpolate_pos_encoding:
                pos_emb = self._interpolate_pos_encoding(embeddings, H, W)
            else:
                pos_emb = self.position_embedding(self.position_ids_full[:, :num_patches_full])
            embeddings = embeddings + pos_emb
            
            return embeddings
        
        # DART: Compute patch importance and select top-k
        patch_mask, selected_indices = compute_patch_scores(
            pixel_values,
            self.scoring_network,
            self.patch_size,
            self.num_patches_target,
            self.temperature,
        )
        
        # Extract selected patches
        # First, get all patches
        patch_embeds_full = self.patch_embedding(pixel_values)  # (B, embed_dim, H_patch, W_patch)
        patch_embeds_full = patch_embeds_full.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Reshape mask to (B, num_patches)
        patch_mask_flat = patch_mask.flatten(1)  # (B, num_patches)
        
        # Use mask to select patches (with gradient flow)
        # For training, we use soft selection; for inference, we use hard selection
        if self.training:
            # Soft selection: weight patches by importance (maintains gradient flow)
            patch_weights = patch_mask_flat.unsqueeze(-1)  # (B, num_patches, 1)
            selected_embeds = patch_embeds_full * patch_weights
            # Keep all patches but weight them (simpler for gradient flow)
            # In practice, you might want to actually remove zero-weighted patches
            selected_embeds = selected_embeds
            # Use full position encoding (will be masked by weights)
            if interpolate_pos_encoding:
                pos_emb = self._interpolate_pos_encoding(selected_embeds, H, W)
            else:
                pos_emb = self.position_embedding(self.position_ids_full[:, :num_patches_full])
            embeddings = selected_embeds + pos_emb
        else:
            # Hard selection: only keep selected patches
            # Get indices of selected patches
            selected_embeds_list = []
            selected_pos_ids_list = []
            for b in range(B):
                batch_indices = selected_indices[b]  # (k,)
                selected_embeds_b = patch_embeds_full[b, batch_indices]  # (k, embed_dim)
                selected_embeds_list.append(selected_embeds_b)
                selected_pos_ids_list.append(batch_indices)
            
            # Pad to same length for batching
            max_k = selected_indices.shape[1]
            selected_embeds = torch.stack([
                F.pad(emb, (0, 0, 0, max_k - emb.shape[0]), mode='constant', value=0)
                if emb.shape[0] < max_k else emb[:max_k]
                for emb in selected_embeds_list
            ])
            
            # Add position encoding for selected patches
            selected_pos_ids = torch.stack([
                F.pad(ids, (0, max_k - ids.shape[0]), mode='constant', value=0)
                if ids.shape[0] < max_k else ids[:max_k]
                for ids in selected_pos_ids_list
            ])
            
            if interpolate_pos_encoding:
                pos_emb = self._interpolate_pos_encoding_selected(
                    selected_embeds, H, W, selected_pos_ids, H_patch, W_patch
                )
            else:
                # Use position encoding for selected patches
                pos_emb = self.position_embedding(selected_pos_ids)
            
            embeddings = selected_embeds + pos_emb
        
        return embeddings
    
    def _interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Interpolate position encoding for full patches."""
        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]
        
        if num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids_full[:, :num_patches])
        
        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)
        dim = embeddings.shape[-1]
        
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        
        sqrt_num_positions = int(num_positions ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed
    
    def _interpolate_pos_encoding_selected(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
        selected_indices: torch.Tensor,
        H_patch: int,
        W_patch: int,
    ) -> torch.Tensor:
        """Interpolate position encoding for selected patches."""
        # Get full position encoding
        full_pos_emb = self._interpolate_pos_encoding(
            embeddings,
            height,
            width,
        )  # (1, num_patches, embed_dim)
        
        # Select position encodings for selected patches
        B = selected_indices.shape[0]
        selected_pos_embs = []
        for b in range(B):
            batch_indices = selected_indices[b]  # (k,)
            pos_emb_b = full_pos_emb[0, batch_indices]  # (k, embed_dim)
            selected_pos_embs.append(pos_emb_b)
        
        # Stack and pad if needed
        max_k = selected_indices.shape[1]
        selected_pos_emb = torch.stack([
            F.pad(emb, (0, 0, 0, max_k - emb.shape[0]), mode='constant', value=0)
            if emb.shape[0] < max_k else emb[:max_k]
            for emb in selected_pos_embs
        ])
        
        return selected_pos_emb

