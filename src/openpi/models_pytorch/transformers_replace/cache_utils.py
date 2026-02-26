# coding=utf-8
# Cache utilities for transformers_replace
from typing import Optional, Tuple
import torch


class Cache:
    """Base class for key-value cache."""
    
    def get_seq_length(self) -> int:
        """Get the sequence length of the cache."""
        raise NotImplementedError
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with new key and value states."""
        raise NotImplementedError


class DynamicCache(Cache):
    """Dynamic cache that grows as needed."""
    
    def __init__(self):
        self.key_cache: list = []
        self.value_cache: list = []
    
    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the key and value cache for a specific layer."""
        if layer_idx < len(self.key_cache):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        return None, None
    
    def __iter__(self):
        """Iterate over the cache."""
        for layer_idx in range(len(self.key_cache)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])
    
    def __len__(self):
        """Get the number of layers in the cache."""
        return len(self.key_cache)
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Get the sequence length of the cache for a specific layer."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with new key and value states."""
        # Ensure the cache has enough layers
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
        
        # If this is the first update for this layer, initialize the cache
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Concatenate new states with existing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], ...]:
        """Convert to legacy cache format."""
        return tuple((self.key_cache[i], self.value_cache[i]) for i in range(len(self.key_cache)))











