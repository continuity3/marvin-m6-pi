import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
import openpi.models_pytorch.tome_pytorch as _tome


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Prepare ToMe configuration for ViT
        tome_config = None
        if hasattr(config, 'tome_enabled') and config.tome_enabled:
            tome_config = {
                "enabled": config.tome_enabled,
                "ratio": config.tome_ratio,
                "metric": config.tome_metric,
                "interval": getattr(config, 'tome_interval', 1),  # Default: every layer (interval=1)
            }
        
        # Prepare ToFu configuration for ViT
        tofu_config = None
        if hasattr(config, 'tofu_enabled') and config.tofu_enabled:
            tofu_config = {
                "enabled": config.tofu_enabled,
                "ratio": config.tofu_ratio,
                "method": config.tofu_method,
                "use_fusion": config.tofu_use_fusion,
                "fusion_ratio": config.tofu_fusion_ratio,
                "interval": getattr(config, 'tofu_interval', 1),  # Default: every layer (interval=1)
            }
        
        # Prepare V2Drop configuration for LLM
        # Always pass config (even if disabled) so the model knows V2Drop is available
        v2drop_config = {
            "enabled": getattr(config, 'v2drop_enabled', False),
            "ratio": getattr(config, 'v2drop_ratio', 0.5),
            "method": getattr(config, 'v2drop_method', 'l2'),
            "interval": getattr(config, 'v2drop_interval', 1),
            "min_tokens": getattr(config, 'v2drop_min_tokens', 1),
        }
        
        # Prepare SnapKV configuration for KV cache compression
        snapkv_config = {
            "enabled": getattr(config, 'snapkv_enabled', False),
            "compression_ratio": getattr(config, 'snapkv_compression_ratio', 0.5),
            "observation_window": getattr(config, 'snapkv_observation_window', 32),
            "clustering_method": getattr(config, 'snapkv_clustering_method', 'topk'),
        }
        
        # Prepare LeanK configuration for K cache channel pruning
        leank_config = {
            "enabled": getattr(config, 'leank_enabled', False),
            "pruning_ratio": getattr(config, 'leank_pruning_ratio', 0.5),
            "method": getattr(config, 'leank_method', 'magnitude'),
            "topk": getattr(config, 'leank_topk', True),
        }
        
        # Prepare DART configuration for adaptive patch selection
        dart_config = {
            "enabled": getattr(config, 'dart_enabled', False),
            "num_patches": getattr(config, 'dart_num_patches', 196),
            "scoring_backbone": getattr(config, 'dart_scoring_backbone', 'mobilenet_v3_small'),
            "temperature": getattr(config, 'dart_temperature', 1.0),
        }
        
        # Store SparseVLM configuration
        self.sparsevlm_config = {
            "enabled": getattr(config, 'sparsevlm_enabled', False),
            "num_retain": getattr(config, 'sparsevlm_num_retain', 192),
            "method": getattr(config, 'sparsevlm_method', 'cross_attention'),
            "version": getattr(config, 'sparsevlm_version', '1.5'),
        }
        
        # Try to import SparseVLM module
        try:
            from openpi.models_pytorch import sparsevlm_pytorch as _sparsevlm_module
            self._sparsevlm_module = _sparsevlm_module
            self._sparsevlm_available = True
        except ImportError:
            self._sparsevlm_module = None
            self._sparsevlm_available = False
            print("[SparseVLM] ⚠️ SparseVLM module not available")
        
        # Log SparseVLM status
        if self.sparsevlm_config.get("enabled", False) and self._sparsevlm_available:
            msg = (
                f"[SparseVLM] ✅ Enabled: num_retain={self.sparsevlm_config.get('num_retain', 192)}, "
                f"method={self.sparsevlm_config.get('method', 'cross_attention')}, "
                f"version={self.sparsevlm_config.get('version', '1.5')}"
            )
            print(msg)
        elif self.sparsevlm_config.get("enabled", False) and not self._sparsevlm_available:
            print("[SparseVLM] ⚠️ SparseVLM disabled (module not available)")
        
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
            tome_config=tome_config,
            tofu_config=tofu_config,
            v2drop_config=v2drop_config,
            snapkv_config=snapkv_config,
            leank_config=leank_config,
            dart_config=dart_config,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        total_img_tokens = 0
        img_token_list = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            
            # Note: ToMe is now applied inside ViT (SiglipEncoder) layer-by-layer
            # This provides better acceleration as tokens are reduced progressively
            # through the ViT layers, not just at the output
            
            img_token_list.append(num_img_embs)
            total_img_tokens += num_img_embs

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        # Apply SparseVLM: sparsify vision tokens based on text guidance
        if (self.sparsevlm_config is not None 
            and self.sparsevlm_config.get("enabled", False) 
            and self._sparsevlm_available
            and len(embs) > 0):  # Only if we have vision tokens
            
            # Concatenate all vision tokens (before language tokens)
            vision_embs = torch.cat(embs, dim=1)  # [B, total_vision_tokens, D]
            num_vision_tokens = vision_embs.shape[1]
            
            # Apply SparseVLM sparsification
            sparsified_vision, keep_mask = self._sparsevlm_module.apply_sparsevlm(
                vision_embs,
                lang_emb,
                num_retain=self.sparsevlm_config.get("num_retain", 192),
                method=self.sparsevlm_config.get("method", "cross_attention"),
                version=self.sparsevlm_config.get("version", "1.5"),
                enabled=True,
            )
            
            # Update embs: replace all vision embs with sparsified version
            embs = [sparsified_vision]
            
            # Update pad_masks: keep only selected vision tokens
            vision_pad_masks = torch.cat(pad_masks, dim=1)  # [B, total_vision_tokens]
            # Use keep_mask to select which vision tokens to keep
            batch_indices = torch.arange(vision_pad_masks.shape[0], device=vision_pad_masks.device)[:, None]
            vision_pad_masks_sparse = vision_pad_masks[batch_indices, keep_mask]  # [B, num_retain]
            pad_masks = [vision_pad_masks_sparse]
            
            # Update att_masks: adjust for reduced vision tokens
            num_retain = sparsified_vision.shape[1]
            att_masks = [0] * num_retain  # Reset att_masks for sparsified vision tokens
            
            print(f"[SparseVLM] ✅ Vision tokens sparsified: {num_vision_tokens} → {num_retain} tokens "
                  f"(reduction={(1-num_retain/num_vision_tokens)*100:.1f}%)")
        
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, total_img_tokens
        
    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    # def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
    #     """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
    #     bsize = observation.state.shape[0]
    #     if noise is None:
    #         actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
    #         noise = self.sample_noise(actions_shape, device)

    #     images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

    #     prefix_embs, prefix_pad_masks, prefix_att_masks, total_img_tokens = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    #     prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    #     prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    #     # Compute image and language key value cache
    #     prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
    #     self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    #     _, past_key_values = self.paligemma_with_expert.forward(
    #         attention_mask=prefix_att_2d_masks_4d,
    #         position_ids=prefix_position_ids,
    #         past_key_values=None,
    #         inputs_embeds=[prefix_embs, None],
    #         use_cache=True,
    #     )

    #     dt = -1.0 / num_steps
    #     dt = torch.tensor(dt, dtype=torch.float32, device=device)

    #     x_t = noise
    #     time = torch.tensor(1.0, dtype=torch.float32, device=device)
    #     while time >= -dt / 2:
    #         expanded_time = time.expand(bsize)
    #         v_t = self.denoise_step(
    #             state,
    #             prefix_pad_masks,
    #             past_key_values,
    #             x_t,
    #             expanded_time,
    #         )

    #         # Euler step - use new tensor assignment instead of in-place operation
    #         x_t = x_t + dt * v_t
    #         time += dt
    #     return x_t
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Inference with detailed latency profiling (ViT / LLM / Flow-Matching)."""

        class CUDATimer:
            def __init__(self):
                self.start = torch.cuda.Event(enable_timing=True)
                self.end = torch.cuda.Event(enable_timing=True)

            def __enter__(self):
                torch.cuda.synchronize()
                self.start.record()
                return self

            def __exit__(self, *args):
                self.end.record()
                torch.cuda.synchronize()

            def elapsed_ms(self):
                return self.start.elapsed_time(self.end)

        timings = {}

        # =========================
        # 0) Prepare noise & input
        # =========================
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = \
            self._preprocess_observation(observation, train=False)

        # =========================
        # 1) ViT (SigLIP) - 只编码图像
        # =========================
        vit_embs = []
        vit_pad_masks = []
        vit_att_masks = []
        total_img_tokens = 0
        
        with CUDATimer() as t:
            for img, img_mask in zip(images, img_masks, strict=True):
                def image_embed_func(img):
                    return self.paligemma_with_expert.embed_image(img)
                
                img_emb = self._apply_checkpoint(image_embed_func, img)
                bsize, num_img_embs = img_emb.shape[:2]
                
                # Note: ToMe is applied inside ViT (SiglipEncoder) layer-by-layer
                # No need to apply again here - the token reduction already happened in ViT
                # Each image is processed separately through ViT, so ToMe is applied per image
                total_img_tokens += num_img_embs
                vit_embs.append(img_emb)
                vit_pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                vit_att_masks += [0] * num_img_embs
        
        # Check total token count after processing all images
        # Check ToMe, ToFu, and V2Drop
        has_tome = hasattr(self.config, 'tome_enabled') and self.config.tome_enabled
        has_tofu = hasattr(self.config, 'tofu_enabled') and self.config.tofu_enabled
        has_v2drop = hasattr(self.config, 'v2drop_enabled') and self.config.v2drop_enabled
        
        if has_tome or has_tofu or has_v2drop:
            # Expected total token count after ToMe/ToFu: num_images * per_image_tokens
            # For SigLIP So400m/14: 224x224 image -> 16x16 patches = 256 tokens per image
            # With 3 images: 3 * 256 = 768 tokens (without optimization)
            num_images = len(images)
            original_total = num_images * 256
            
            # Calculate expected tokens based on enabled optimization
            # Note: V2Drop is applied in LLM layers, not in ViT, so initial token count
            # will be the same, but will reduce during LLM forward pass
            if has_tofu:
                # ToFu: filter to ratio, then optionally fuse
                filtered_per_image = int(256 * self.config.tofu_ratio)
                if hasattr(self.config, 'tofu_use_fusion') and self.config.tofu_use_fusion:
                    expected_per_image = int(filtered_per_image * self.config.tofu_fusion_ratio)
                else:
                    expected_per_image = filtered_per_image
                expected_total = num_images * expected_per_image
                method_name = "ToFu"
            elif has_tome:
                # ToMe: merge tokens
                expected_per_image = int(256 * self.config.tome_ratio) if self.config.tome_ratio < 0.5 else 128
                expected_total = num_images * expected_per_image
                method_name = "ToMe"
            elif has_v2drop:
                # V2Drop: applied in LLM, initial tokens unchanged
                expected_total = original_total
                method_name = "V2Drop (will reduce in LLM)"
            else:
                expected_total = original_total
                method_name = "None"
            
            tokens_per_image = total_img_tokens // num_images if num_images > 0 else 0
            reduction_pct = (1.0 - total_img_tokens / original_total) * 100 if original_total > 0 else 0.0
            
            if total_img_tokens < original_total * 0.8:  # At least 20% reduction
                print(f"[{method_name} Check] ✅ Total image tokens reduced: {total_img_tokens} tokens ({num_images} images × {tokens_per_image} tokens/image, expected ~{expected_total}, reduction={reduction_pct:.1f}%)")
            else:
                print(f"[{method_name} Check] ⚠️ Total image tokens may not be reduced: {total_img_tokens} tokens ({num_images} images × {tokens_per_image} tokens/image, expected ~{expected_total}, reduction={reduction_pct:.1f}%)")
        
        timings["vit"] = t.elapsed_ms()
        
        # =========================
        # 2) Text embed (语言 token 编码)
        # =========================
        with CUDATimer() as t:
            def lang_embed_func(lang_tokens):
                lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
                lang_emb_dim = lang_emb.shape[-1]
                return lang_emb * math.sqrt(lang_emb_dim)
            
            lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
            num_lang_embs = lang_emb.shape[1]
            vit_embs.append(lang_emb)
            vit_pad_masks.append(lang_masks)
            vit_att_masks += [0] * num_lang_embs
            
            prefix_embs = torch.cat(vit_embs, dim=1)
            prefix_pad_masks = torch.cat(vit_pad_masks, dim=1)
            prefix_att_masks = torch.tensor(vit_att_masks, dtype=torch.bool, device=prefix_pad_masks.device)
            bsize = prefix_pad_masks.shape[0]
            prefix_att_masks = prefix_att_masks[None, :].expand(bsize, len(vit_att_masks))
        timings["text_embed"] = t.elapsed_ms()

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # =========================
        # 3) LLM prefix forward (KV cache)
        # =========================
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        with CUDATimer() as t:
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        timings["llm_prefix"] = t.elapsed_ms()

        # =========================
        # 4) Flow Matching (ODE loop)
        # =========================
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        flow_llm_total = 0.0
        flow_other_total = 0.0
        steps = 0

        # 计时 Flow Matching 中的非 LLM 部分（Euler update 等）
        with CUDATimer() as t_flow:
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)

                # ---- single denoise step (Expert LLM) ----
                with CUDATimer() as t_step:
                    v_t = self.denoise_step(
                        state,
                        prefix_pad_masks,
                        past_key_values,
                        x_t,
                        expanded_time,
                    )
                flow_llm_total += t_step.elapsed_ms()
                steps += 1

                # Euler update (非 LLM 部分)
                with CUDATimer() as t_euler:
                    x_t = x_t + dt * v_t
                    time += dt
                flow_other_total += t_euler.elapsed_ms()

        timings["flow_matching_total"] = t_flow.elapsed_ms()
        timings["flow_llm_total"] = flow_llm_total
        timings["flow_llm_avg"] = flow_llm_total / max(steps, 1)
        timings["flow_other"] = flow_other_total
        timings["flow_steps"] = steps

        # =========================
        # 5) 计算 LLM 总时间
        # =========================
        timings["llm_total"] = timings["llm_prefix"] + timings["flow_llm_total"]
        
        # =========================
        # 6) Total
        # =========================
        timings["total"] = (
            timings["vit"]
            + timings["text_embed"]
            + timings["llm_total"]
            + timings["flow_other"]
        )

        # =========================
        # 7) Print summary
        # =========================
        print(
            f"\n[PI0 Inference Profiling]\n"
            f"  ViT (SigLIP)            : {timings['vit']:.2f} ms\n"
            f"  Text Embed             : {timings['text_embed']:.2f} ms\n"
            f"  LLM (total)            : {timings['llm_total']:.2f} ms\n"
            f"    ├─ prefix (KV cache): {timings['llm_prefix']:.2f} ms\n"
            f"    └─ flow matching    : {timings['flow_llm_total']:.2f} ms "
            f"(avg {timings['flow_llm_avg']:.2f} ms/step, {timings['flow_steps']} steps)\n"
            f"  Flow Matching (other)  : {timings['flow_other']:.2f} ms\n"
            f"  --------------------------------------\n"
            f"  Total inference        : {timings['total']:.2f} ms\n"
        )

        return x_t
    # def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
    #     import time
    #     timings = {}
    #     torch.cuda.synchronize()
    #     total_t0 = time.perf_counter()
    #     t0 = time.perf_counter()
    #     bsize = observation.state.shape[0]
    #     if noise is None:
    #         actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
    #         noise = self.sample_noise(actions_shape, device)

    #     images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
    #     prefix_embs, prefix_pad_masks, prefix_att_masks, total_img_tokens = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    #     # siglip_tokens = sum([m.sum().item() for m in img_masks])
    #     # prefix_embs, prefix_pad_masks, prefix_att_masks = \
    #     #     self.paligemma_with_expert.prune_vision_tokens(
    #     #         prefix_embs,
    #     #         prefix_pad_masks,
    #     #         prefix_att_masks,
    #     #         img_tokens=total_img_tokens,
    #     #         prune_ratio=0.50,
    #     #     )
    #     prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    #     prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    #     torch.cuda.synchronize()
    #     timings["embed_prefix"] = (time.perf_counter() - t0) * 1000

    #     t0 = time.perf_counter()
    #     prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
    #     self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    #     _, past_key_values = self.paligemma_with_expert.forward(
    #         attention_mask=prefix_att_2d_masks_4d,
    #         position_ids=prefix_position_ids,
    #         past_key_values=None,
    #         inputs_embeds=[prefix_embs, None],
    #         use_cache=True,
    #     )
    #     torch.cuda.synchronize()
    #     timings["prefix_forward"] = (time.perf_counter() - t0) * 1000
    #     t0 = time.perf_counter()
    #     dt = -1.0 / num_steps
    #     dt = torch.tensor(dt, dtype=torch.float32, device=device)
    #     x_t = noise
    #     time_now = torch.tensor(1.0, dtype=torch.float32, device=device)

    #     v_prev = [None]  
    #     pred_interval = 7
    #     step = 0
    #     while time_now >= -dt / 2:
    #         expanded_time = time_now.expand(bsize)

    #         # === O = 0 predictor ===
    #         if (v_prev[0] is not None) and (step % pred_interval != 0):
    #             v_t = v_prev[0]  
    #             mode = "forecast"
    #         else:
    #             # === True forward ===
    #             v_t = self.denoise_step(
    #                 state,
    #                 prefix_pad_masks,
    #                 past_key_values,
    #                 x_t,
    #                 expanded_time,
    #             )
    #             mode = "true"

    #         # Euler 更新
    #         x_t = x_t + dt * v_t

    #         # 更新缓存（只保存1个）
    #         v_prev = [v_t]

    #         if step % pred_interval == 0:
    #             print(f"[TaylorSeer] Step {step:02d} → True Forward")
    #         else:
    #             print(f"[TaylorSeer] Step {step:02d} → Forecasted (Predicted, O=0)")

    #         time_now += dt
    #         step += 1

    #     torch.cuda.synchronize()
    #     timings["denoise"] = (time.perf_counter() - t0) * 1000

    #     # === 4️⃣ 汇总并打印 ===
    #     total_time = (time.perf_counter() - total_t0) * 1000
    #     timings["total"] = total_time
    #     print(f"[⏱ TaylorSeer-O1] EmbedPrefix: {timings['embed_prefix']:.2f} ms | "
    #         f"PrefixForward: {timings['prefix_forward']:.2f} ms | "
    #         f"Denoise: {timings['denoise']:.2f} ms | Total: {timings['total']:.2f} ms")

    #     return x_t
    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
