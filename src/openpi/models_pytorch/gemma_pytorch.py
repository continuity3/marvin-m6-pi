from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        tome_config=None,  # ToMe configuration dict: {enabled, ratio, metric}
        tofu_config=None,  # ToFu configuration dict: {enabled, ratio, method, use_fusion, fusion_ratio}
        v2drop_config=None,  # V2Drop configuration dict: {enabled, ratio, method, interval, min_tokens}
        snapkv_config=None,  # SnapKV configuration dict: {enabled, compression_ratio, observation_window, clustering_method}
        leank_config=None,  # LeanK configuration dict: {enabled, pruning_ratio, method, topk}
        dart_config=None,  # DART configuration dict: {enabled, num_patches, scoring_backbone, temperature}
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"
        
        # Pass ToMe configuration to vision_config
        if tome_config is not None:
            # Use setattr to ensure attributes are set even if config object doesn't support direct assignment
            setattr(vlm_config_hf.vision_config, 'tome_enabled', tome_config.get("enabled", False))
            setattr(vlm_config_hf.vision_config, 'tome_ratio', tome_config.get("ratio", 0.75))
            setattr(vlm_config_hf.vision_config, 'tome_metric', tome_config.get("metric", "cosine"))
            setattr(vlm_config_hf.vision_config, 'tome_interval', tome_config.get("interval", 1))
        
        # Pass ToFu configuration to vision_config
        if tofu_config is not None:
            setattr(vlm_config_hf.vision_config, 'tofu_enabled', tofu_config.get("enabled", False))
            setattr(vlm_config_hf.vision_config, 'tofu_ratio', tofu_config.get("ratio", 0.75))
            setattr(vlm_config_hf.vision_config, 'tofu_method', tofu_config.get("method", "norm"))
            setattr(vlm_config_hf.vision_config, 'tofu_use_fusion', tofu_config.get("use_fusion", True))
            setattr(vlm_config_hf.vision_config, 'tofu_fusion_ratio', tofu_config.get("fusion_ratio", 0.5))
            setattr(vlm_config_hf.vision_config, 'tofu_interval', tofu_config.get("interval", 1))
        
        # Pass DART configuration to vision_config
        if dart_config is not None:
            setattr(vlm_config_hf.vision_config, 'dart_enabled', dart_config.get("enabled", False))
            setattr(vlm_config_hf.vision_config, 'dart_num_patches', dart_config.get("num_patches", 196))
            setattr(vlm_config_hf.vision_config, 'dart_scoring_backbone', dart_config.get("scoring_backbone", "mobilenet_v3_small"))
            setattr(vlm_config_hf.vision_config, 'dart_temperature', dart_config.get("temperature", 1.0))
            if dart_config.get("enabled", False):
                print(
                    f"[DART] ✅ Enabled: num_patches={dart_config.get('num_patches', 196)}, "
                    f"scoring_backbone={dart_config.get('scoring_backbone', 'mobilenet_v3_small')}"
                )
        
        # Store V2Drop configuration
        self.v2drop_config = v2drop_config
        # Always try to import V2Drop module (for availability check)
        try:
            import openpi.models_pytorch.v2drop_pytorch as _v2drop_module
            self._v2drop_module = _v2drop_module
            self._v2drop_available = True
        except ImportError:
            self._v2drop_module = None
            self._v2drop_available = False
            print("[V2Drop] ⚠️ V2Drop module not available")
        
        # Log status based on config
        if v2drop_config is not None:
            if v2drop_config.get("enabled", False) and self._v2drop_available:
                msg = (
                    f"[V2Drop] ✅ Enabled in LLM: drop_ratio={v2drop_config.get('ratio', 0.5):.3f}, "
                    f"method={v2drop_config.get('method', 'l2')}, "
                    f"interval={v2drop_config.get('interval', 1)}, "
                    f"min_tokens={v2drop_config.get('min_tokens', 1)}"
                )
                print(msg)
            elif not v2drop_config.get("enabled", False):
                print(f"[V2Drop] ⚠️ V2Drop disabled in config (enabled=False)")
            elif not self._v2drop_available:
                print("[V2Drop] ⚠️ V2Drop module not available")
        
        # Store SnapKV configuration
        self.snapkv_config = snapkv_config
        # Always try to import SnapKV module
        try:
            from openpi.models_pytorch import snapkv_pytorch as _snapkv_module
            self._snapkv_module = _snapkv_module
            self._snapkv_available = True
        except ImportError:
            self._snapkv_module = None
            self._snapkv_available = False
            print("[SnapKV] ⚠️ SnapKV module not available")
        
        # Log SnapKV status
        if snapkv_config is not None:
            if snapkv_config.get("enabled", False) and self._snapkv_available:
                msg = (
                    f"[SnapKV] ✅ Enabled: compression_ratio={snapkv_config.get('compression_ratio', 0.5):.3f}, "
                    f"observation_window={snapkv_config.get('observation_window', 32)}, "
                    f"clustering_method={snapkv_config.get('clustering_method', 'topk')}"
                )
                print(msg)
            elif not snapkv_config.get("enabled", False):
                print(f"[SnapKV] ⚠️ SnapKV disabled in config (enabled=False)")
            elif not self._snapkv_available:
                print("[SnapKV] ⚠️ SnapKV module not available")
        
        # Store LeanK configuration
        self.leank_config = leank_config
        # Always try to import LeanK module
        try:
            from openpi.models_pytorch import leank_pytorch as _leank_module
            self._leank_module = _leank_module
            self._leank_available = True
        except ImportError:
            self._leank_module = None
            self._leank_available = False
            print("[LeanK] ⚠️ LeanK module not available")
        
        # Log LeanK status
        if leank_config is not None:
            if leank_config.get("enabled", False) and self._leank_available:
                msg = (
                    f"[LeanK] ✅ Enabled: pruning_ratio={leank_config.get('pruning_ratio', 0.5):.3f}, "
                    f"method={leank_config.get('method', 'magnitude')}, "
                    f"topk={leank_config.get('topk', True)}"
                )
                print(msg)
            elif not leank_config.get("enabled", False):
                print(f"[LeanK] ⚠️ LeanK disabled in config (enabled=False)")
            elif not self._leank_available:
                print("[LeanK] ⚠️ LeanK module not available")

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            # Prefix forward: manually process layers to apply V2Drop
            if (self._v2drop_available 
                and self.v2drop_config is not None 
                and self.v2drop_config.get("enabled", False)):
                # Manual layer-by-layer processing with V2Drop
                hidden_states = inputs_embeds[0]
                # In prefix, tokens are: vision tokens + language tokens
                # We need to estimate num_vision_tokens from the total sequence length
                # For SigLIP: 3 images × 256 tokens = 768 vision tokens (typical case)
                # Language tokens are typically 10-100 tokens
                total_tokens = hidden_states.shape[1]
                
                # Heuristic: estimate vision tokens based on typical patterns
                # Common cases:
                # - 3 images × 256 = 768 vision tokens + ~20-50 lang tokens = ~790-820 total
                # - 1 image × 256 = 256 vision tokens + ~20-50 lang tokens = ~280-310 total
                if total_tokens >= 700:
                    # Likely 3 images: estimate vision tokens as total - typical_lang_tokens
                    num_vision_tokens = max(1, total_tokens - 50)  # Assume ~50 lang tokens
                elif total_tokens >= 250:
                    # Likely 1 image: estimate vision tokens as total - typical_lang_tokens
                    num_vision_tokens = max(1, total_tokens - 50)  # Assume ~50 lang tokens
                else:
                    # For smaller sequences, assume all are vision tokens
                    num_vision_tokens = total_tokens
                
                print(f"[V2Drop] Prefix forward: total_tokens={total_tokens}, estimated num_vision_tokens={num_vision_tokens}")
                num_layers = self.paligemma.config.text_config.num_hidden_layers
                
                # Initialize past_key_values if use_cache and not provided
                if use_cache and past_key_values is None:
                    from transformers.cache_utils import DynamicCache
                    past_key_values = DynamicCache()
                
                # Compute position_embeddings once (used by all layers)
                position_embeddings = self.paligemma.language_model.rotary_emb(hidden_states, position_ids)
                
                # Compute cache_position if needed
                if use_cache and past_key_values is not None:
                    cache_position = torch.arange(
                        past_key_values.get_seq_length(),
                        past_key_values.get_seq_length() + hidden_states.shape[1],
                        device=hidden_states.device
                    )
                else:
                    cache_position = None
                
                for layer_idx in range(num_layers):
                    tokens_before = hidden_states.clone()
                    
                    # Process one layer
                    layer = self.paligemma.language_model.layers[layer_idx]
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,  # Pass the Cache object, not a list element
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,  # Pass pre-computed position embeddings
                        adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
                        output_attentions=False,  # We don't need attention weights
                    )
                    # layer_outputs is a tuple: (hidden_states,) or (hidden_states, self_attn_weights)
                    hidden_states = layer_outputs[0]
                    
                    # Apply V2Drop after each layer (only in first 5 layers)
                    if layer_idx < 5 and layer_idx % self.v2drop_config.get("interval", 1) == 0:
                        tokens_after = hidden_states
                        original_num_tokens = tokens_after.shape[1]
                        original_num_vision = num_vision_tokens
                        
                        filtered_tokens, keep_mask = self._v2drop_module.apply_v2drop(
                            tokens_before=tokens_before,
                            tokens_after=tokens_after,
                            num_vision_tokens=num_vision_tokens,
                            drop_ratio=self.v2drop_config.get("ratio", 0.5),
                            method=self.v2drop_config.get("method", "l2"),
                            enabled=True,
                            min_tokens=self.v2drop_config.get("min_tokens", 1),
                        )
                        
                        new_num_tokens = filtered_tokens.shape[1]
                        new_num_vision = keep_mask[:, :num_vision_tokens].sum(dim=1).min().item()
                        
                        hidden_states = filtered_tokens
                        num_vision_tokens = new_num_vision
                        
                        # Print debug info if tokens were dropped
                        if new_num_tokens < original_num_tokens:
                            reduction_pct = (1.0 - new_num_tokens / original_num_tokens) * 100
                            vision_reduction_pct = (1.0 - new_num_vision / original_num_vision) * 100 if original_num_vision > 0 else 0.0
                            msg = (
                                f"[V2Drop] LLM Layer {layer_idx+1}: {original_num_tokens} -> {new_num_tokens} tokens "
                                f"(vision: {original_num_vision} -> {new_num_vision}, "
                                f"total reduction={reduction_pct:.1f}%, vision reduction={vision_reduction_pct:.1f}%)"
                            )
                            print(msg)
                        
                        # Update attention_mask and position_ids
                        if attention_mask is not None:
                            new_seq_len = filtered_tokens.shape[1]
                            if attention_mask.shape[-1] > new_seq_len:
                                attention_mask = attention_mask[:, :new_seq_len]
                            if attention_mask.dim() == 4 and attention_mask.shape[-2] > new_seq_len:
                                attention_mask = attention_mask[:, :, :new_seq_len, :new_seq_len]
                        
                        if position_ids is not None:
                            new_seq_len = filtered_tokens.shape[1]
                            if position_ids.shape[1] > new_seq_len:
                                position_ids = position_ids[:, :new_seq_len]
                        
                        # Recompute position_embeddings and cache_position for next layer (token count changed)
                        position_embeddings = self.paligemma.language_model.rotary_emb(hidden_states, position_ids)
                        if use_cache and past_key_values is not None:
                            cache_position = torch.arange(
                                past_key_values.get_seq_length(),
                                past_key_values.get_seq_length() + hidden_states.shape[1],
                                device=hidden_states.device
                            )
                
                # Final norm
                hidden_states = self.paligemma.language_model.norm(hidden_states)
                
                # Create output object matching transformers format
                from transformers.modeling_outputs import BaseModelOutputWithPast
                prefix_output = BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=past_key_values,
                )
                prefix_past_key_values = prefix_output.past_key_values
                
                # Apply SnapKV compression to KV cache after prefix forward
                if (self.snapkv_config is not None 
                    and self.snapkv_config.get("enabled", False) 
                    and self._snapkv_available
                    and prefix_past_key_values is not None
                    and use_cache):
                    # Debug: log before compression
                    if hasattr(prefix_past_key_values, 'key_cache') and len(prefix_past_key_values.key_cache) > 0:
                        original_seq_len = prefix_past_key_values.key_cache[0].shape[-2]
                        print(f"[SnapKV] 🔄 Compressing KV cache: original_seq_len={original_seq_len}")
                    
                    prefix_past_key_values = self._snapkv_module.apply_snapkv_to_cache(
                        prefix_past_key_values,
                        compression_ratio=self.snapkv_config.get("compression_ratio", 0.5),
                        observation_window=self.snapkv_config.get("observation_window", 32),
                        clustering_method=self.snapkv_config.get("clustering_method", "topk"),
                        enabled=True,
                    )
                    
                    # Debug: log after compression
                    if hasattr(prefix_past_key_values, 'key_cache') and len(prefix_past_key_values.key_cache) > 0:
                        compressed_seq_len = prefix_past_key_values.key_cache[0].shape[-2]
                        compression_ratio_actual = compressed_seq_len / original_seq_len if original_seq_len > 0 else 1.0
                        print(f"[SnapKV] ✅ KV cache compressed: {original_seq_len} → {compressed_seq_len} tokens "
                              f"(ratio={compression_ratio_actual:.3f}, reduction={(1-compression_ratio_actual)*100:.1f}%)")
                
                # Apply LeanK channel pruning to K cache after SnapKV (if enabled)
                if (self.leank_config is not None 
                    and self.leank_config.get("enabled", False) 
                    and self._leank_available
                    and prefix_past_key_values is not None
                    and use_cache):
                    if hasattr(prefix_past_key_values, 'key_cache') and len(prefix_past_key_values.key_cache) > 0:
                        original_head_dim = prefix_past_key_values.key_cache[0].shape[-1]
                        print(f"[LeanK] 🔄 Pruning K cache channels: original_head_dim={original_head_dim}")
                    
                    prefix_past_key_values = self._leank_module.apply_leank_to_cache(
                        prefix_past_key_values,
                        pruning_ratio=self.leank_config.get("pruning_ratio", 0.5),
                        method=self.leank_config.get("method", "magnitude"),
                        scorer=None,  # For inference, use non-learnable methods
                        topk=self.leank_config.get("topk", True),
                        enabled=True,
                    )
                    
                    # Debug: log after pruning
                    if hasattr(prefix_past_key_values, 'key_cache') and len(prefix_past_key_values.key_cache) > 0:
                        pruned_head_dim = prefix_past_key_values.key_cache[0].shape[-1]
                        pruning_ratio_actual = pruned_head_dim / original_head_dim if original_head_dim > 0 else 1.0
                        print(f"[LeanK] ✅ K cache channels pruned: {original_head_dim} → {pruned_head_dim} dims "
                              f"(ratio={pruning_ratio_actual:.3f}, reduction={(1-pruning_ratio_actual)*100:.1f}%)")
                
                prefix_output = prefix_output.last_hidden_state
                suffix_output = None
            else:
                # Standard forward without V2Drop
                prefix_output = self.paligemma.language_model.forward(
                    inputs_embeds=inputs_embeds[0],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
                )
                prefix_past_key_values = prefix_output.past_key_values
                
                # Apply SnapKV compression to KV cache after prefix forward
                if (self.snapkv_config is not None 
                    and self.snapkv_config.get("enabled", False) 
                    and self._snapkv_available
                    and prefix_past_key_values is not None
                    and use_cache):
                    # Debug: log before compression
                    if hasattr(prefix_past_key_values, 'key_cache') and len(prefix_past_key_values.key_cache) > 0:
                        original_seq_len = prefix_past_key_values.key_cache[0].shape[-2]
                        print(f"[SnapKV] 🔄 Compressing KV cache: original_seq_len={original_seq_len}")
                    
                    prefix_past_key_values = self._snapkv_module.apply_snapkv_to_cache(
                        prefix_past_key_values,
                        compression_ratio=self.snapkv_config.get("compression_ratio", 0.5),
                        observation_window=self.snapkv_config.get("observation_window", 32),
                        clustering_method=self.snapkv_config.get("clustering_method", "topk"),
                        enabled=True,
                    )
                    
                    # Debug: log after compression
                    if hasattr(prefix_past_key_values, 'key_cache') and len(prefix_past_key_values.key_cache) > 0:
                        compressed_seq_len = prefix_past_key_values.key_cache[0].shape[-2]
                        compression_ratio_actual = compressed_seq_len / original_seq_len if original_seq_len > 0 else 1.0
                        print(f"[SnapKV] ✅ KV cache compressed: {original_seq_len} → {compressed_seq_len} tokens "
                              f"(ratio={compression_ratio_actual:.3f}, reduction={(1-compression_ratio_actual)*100:.1f}%)")
                
                # Apply LeanK channel pruning to K cache after SnapKV (if enabled)
                if (self.leank_config is not None 
                    and self.leank_config.get("enabled", False) 
                    and self._leank_available
                    and prefix_past_key_values is not None
                    and use_cache):
                    if hasattr(prefix_past_key_values, 'key_cache') and len(prefix_past_key_values.key_cache) > 0:
                        original_head_dim = prefix_past_key_values.key_cache[0].shape[-1]
                        print(f"[LeanK] 🔄 Pruning K cache channels: original_head_dim={original_head_dim}")
                    
                    prefix_past_key_values = self._leank_module.apply_leank_to_cache(
                        prefix_past_key_values,
                        pruning_ratio=self.leank_config.get("pruning_ratio", 0.5),
                        method=self.leank_config.get("method", "magnitude"),
                        scorer=None,  # For inference, use non-learnable methods
                        topk=self.leank_config.get("topk", True),
                        enabled=True,
                    )
                    
                    # Debug: log after pruning
                    if hasattr(prefix_past_key_values, 'key_cache') and len(prefix_past_key_values.key_cache) > 0:
                        pruned_head_dim = prefix_past_key_values.key_cache[0].shape[-1]
                        pruning_ratio_actual = pruned_head_dim / original_head_dim if original_head_dim > 0 else 1.0
                        print(f"[LeanK] ✅ K cache channels pruned: {original_head_dim} → {pruned_head_dim} dims "
                              f"(ratio={pruning_ratio_actual:.3f}, reduction={(1-pruning_ratio_actual)*100:.1f}%)")
                
                prefix_output = prefix_output.last_hidden_state
                suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Get num_vision_tokens from prefix embeddings (first element of inputs_embeds)
            num_vision_tokens = inputs_embeds[0].shape[1] if inputs_embeds[0] is not None else 0
            
            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                # Store tokens before layer for V2Drop
                tokens_before = inputs_embeds[0].clone() if inputs_embeds[0] is not None else None
                
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )
                
                # Apply V2Drop after each layer (only in first 5 layers, if enabled and interval matches)
                if (self._v2drop_available 
                    and self.v2drop_config is not None 
                    and self.v2drop_config.get("enabled", False)
                    and tokens_before is not None
                    and inputs_embeds[0] is not None
                    and layer_idx < 5  # Only apply in first 5 layers
                    and layer_idx % self.v2drop_config.get("interval", 1) == 0):
                    
                    tokens_after = inputs_embeds[0]
                    original_num_tokens = tokens_after.shape[1]
                    original_num_vision = num_vision_tokens
                    
                    filtered_tokens, keep_mask = self._v2drop_module.apply_v2drop(
                        tokens_before=tokens_before,
                        tokens_after=tokens_after,
                        num_vision_tokens=num_vision_tokens,
                        drop_ratio=self.v2drop_config.get("ratio", 0.5),
                        method=self.v2drop_config.get("method", "l2"),
                        enabled=True,
                        min_tokens=self.v2drop_config.get("min_tokens", 1),
                    )
                    
                    new_num_tokens = filtered_tokens.shape[1]
                    new_num_vision = keep_mask[:, :num_vision_tokens].sum(dim=1).min().item()
                    
                    # Update inputs_embeds with filtered tokens
                    inputs_embeds[0] = filtered_tokens
                    
                    # Update num_vision_tokens based on keep_mask
                    num_vision_tokens = new_num_vision
                    
                    # Print debug info if tokens were dropped
                    if new_num_tokens < original_num_tokens:
                        reduction_pct = (1.0 - new_num_tokens / original_num_tokens) * 100
                        vision_reduction_pct = (1.0 - new_num_vision / original_num_vision) * 100 if original_num_vision > 0 else 0.0
                        msg = (
                            f"[V2Drop] LLM Layer {layer_idx+1}: {original_num_tokens} -> {new_num_tokens} tokens "
                            f"(vision: {original_num_vision} -> {new_num_vision}, "
                            f"total reduction={reduction_pct:.1f}%, vision reduction={vision_reduction_pct:.1f}%)"
                        )
                        print(msg)
                    
                    # Update attention_mask and position_ids if needed
                    if attention_mask is not None:
                        # Adjust attention_mask to match new token count
                        new_seq_len = filtered_tokens.shape[1]
                        if attention_mask.shape[-1] > new_seq_len:
                            attention_mask = attention_mask[:, :new_seq_len]
                        if attention_mask.dim() == 4 and attention_mask.shape[-2] > new_seq_len:
                            attention_mask = attention_mask[:, :, :new_seq_len, :new_seq_len]
                    
                    if position_ids is not None:
                        # Adjust position_ids to match new token count
                        new_seq_len = filtered_tokens.shape[1]
                        if position_ids.shape[1] > new_seq_len:
                            position_ids = position_ids[:, :new_seq_len]

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
