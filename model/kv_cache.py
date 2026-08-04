import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        if self.cache_k is None or self.cache_v is None:
            # First forward pass (Prompt phase)
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            # Subsequent steps (Generation phase): Concatenate along sequence dimension (dim=1)
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
            
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        # 2. If kv_cache is None, create a new KVCache
        # 3. Update the cache with the new K and V
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        # 5. Return (rounded output, kv_cache)

        q_new = self.q_proj(x)
        k_new = self.k_proj(x)
        v_new = self.v_proj(x)

        if kv_cache is None:
            kv_cache = KVCache()
        
        k_full, v_full = kv_cache.update(k_new, v_new)

        d_k = q_new.shape[2]
        scores = torch.matmul(q_new, k_full.transpose(-2, -1)) / (d_k ** 0.5)
        
        q_len, k_len = q_new.shape[1], k_full.shape[1]
        
        # # Apply causal mask ONLY during prompt processing (q_len > 1)
        # if q_len > 1:
        #     mask = torch.tril(torch.ones((q_len, k_len), device=x.device))
        #     scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        output = torch.round(torch.matmul(weights, v_full), decimals=4)

        return output, kv_cache