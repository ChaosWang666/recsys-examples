"""Torch based KV cache utilities for NPU inference.

The cache is implemented with regular :class:`torch.Tensor` objects and
plain Python bookkeeping so that it runs on both CPU and NPU devices
without relying on GPU specific extensions such as TensorRT-LLM,
flashinfer or paged_kvcache_ops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class _LayerCache:
    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None

    def append(self, new_key: torch.Tensor, new_value: torch.Tensor, max_length: int) -> None:
        """Append ``new_key``/``new_value`` to the cache while respecting ``max_length``."""
        if self.key is None:
            self.key = new_key
            self.value = new_value
        else:
            self.key = torch.cat([self.key, new_key], dim=0)
            self.value = torch.cat([self.value, new_value], dim=0)
        if self.key.shape[0] > max_length:
            start = self.key.shape[0] - max_length
            self.key = self.key[start:]
            self.value = self.value[start:]

    def length(self) -> int:
        if self.key is None:
            return 0
        return int(self.key.shape[0])


@dataclass
class UserCache:
    """KV cache for a single user across all layers."""

    num_layers: int
    layers: List[_LayerCache] = field(init=False)

    def __post_init__(self) -> None:
        self.layers = [_LayerCache() for _ in range(self.num_layers)]

    def summary(self) -> Tuple[int, List[int]]:
        lengths = [layer.length() for layer in self.layers]
        start_pos = 0 if lengths else -1
        return start_pos, lengths


class KVCacheManager:
    """Simple torch based cache manager.

    The cache keeps a mapping from ``user_id`` to :class:`UserCache` objects.
    Only torch operators are used so the code runs on NPU devices without
    CUDA dependencies.
    """

    def __init__(self, num_layers: int, hidden_size: int, max_length: int, device: torch.device) -> None:
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.device = device
        self._cache: Dict[int, UserCache] = {}

    def get_user_kvdata_info(self, user_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        starts, lengths = [], []
        for uid in user_ids.tolist():
            user_cache = self._cache.get(int(uid))
            if user_cache is None:
                starts.append(0)
                lengths.append(0)
            else:
                start_pos, layer_lengths = user_cache.summary()
                starts.append(start_pos)
                lengths.append(max(layer_lengths))
        return (
            torch.tensor(starts, device=self.device, dtype=torch.int32),
            torch.tensor(lengths, device=self.device, dtype=torch.int32),
        )

    def _ensure_user(self, uid: int) -> UserCache:
        if uid not in self._cache:
            self._cache[uid] = UserCache(self.num_layers)
        return self._cache[uid]

    def fetch_past_kv(
        self, user_ids: torch.Tensor, layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return padded past KV for a batch.

        Returns a tuple ``(past_k, past_v, padding_mask)`` where padding mask is a
        boolean tensor with shape ``(batch, past_seq_len)``.
        """
        past_keys: List[torch.Tensor] = []
        past_values: List[torch.Tensor] = []
        lengths: List[int] = []
        for uid in user_ids.tolist():
            layer_cache = self._ensure_user(int(uid)).layers[layer_idx]
            if layer_cache.key is None:
                lengths.append(0)
                past_keys.append(None)  # type: ignore
                past_values.append(None)  # type: ignore
            else:
                lengths.append(layer_cache.key.shape[0])
                past_keys.append(layer_cache.key)
                past_values.append(layer_cache.value)
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            return None, None, None

        batch = len(user_ids)
        key_buffer = torch.zeros(max_len, batch, self.hidden_size, device=self.device)
        value_buffer = torch.zeros_like(key_buffer)
        padding = torch.ones(batch, max_len, dtype=torch.bool, device=self.device)
        for idx, (k_tensor, v_tensor, length) in enumerate(zip(past_keys, past_values, lengths)):
            if length == 0 or k_tensor is None or v_tensor is None:
                continue
            key_buffer[:length, idx, :] = k_tensor
            value_buffer[:length, idx, :] = v_tensor
            padding[idx, :length] = False
        return key_buffer, value_buffer, padding

    def append(self, user_ids: torch.Tensor, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Append new KV pairs for the batch."""
        assert keys.shape[:2] == values.shape[:2]
        seq_len, batch = keys.shape[:2]
        for b_idx in range(batch):
            uid = int(user_ids[b_idx])
            user_cache = self._ensure_user(uid)
            user_cache.layers[layer_idx].append(keys[:, b_idx, :], values[:, b_idx, :], self.max_length)

    def evict(self, user_ids: Optional[torch.Tensor] = None) -> None:
        """Remove KV entries for selected users or everything if ``user_ids`` is ``None``."""
        if user_ids is None:
            self._cache.clear()
            return
        for uid in user_ids.tolist():
            self._cache.pop(int(uid), None)

    def __len__(self) -> int:
        return len(self._cache)
