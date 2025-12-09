"""Lightweight HSTU style model for NPU inference.

The implementation focuses on incremental inference with KV cache support
and avoids any GPU specific kernels so it can run on NPU/CPU environments.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KVCacheConfig, ModelConfig
from .kv_cache import KVCacheManager


def _build_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int, device: torch.device):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch, dim)
        length = x.shape[0]
        return x + self.pe[:length].unsqueeze(1)


class HSTULayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, kv_manager: KVCacheManager) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.kv_manager = kv_manager
        self.hidden_size = config.hidden_size
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward_single(self, hidden: torch.Tensor, user_id: int) -> torch.Tensor:
        # hidden shape: (seq_len, hidden)
        q = self.q_proj(hidden)
        k_new = self.k_proj(hidden)
        v_new = self.v_proj(hidden)

        user_tensor = torch.tensor([user_id], device=hidden.device)
        past_k, past_v, past_mask = self.kv_manager.fetch_past_kv(user_tensor, self.layer_idx)
        if past_k is not None and past_v is not None:
            past_k_single = past_k[:, 0, :]
            past_v_single = past_v[:, 0, :]
            if past_mask is not None:
                valid_len = int((~past_mask[0]).sum())
                past_k_single = past_k_single[:valid_len]
                past_v_single = past_v_single[:valid_len]
            k_all = torch.cat([past_k_single, k_new], dim=0)
            v_all = torch.cat([past_v_single, v_new], dim=0)
        else:
            k_all, v_all = k_new, v_new

        seq_len = hidden.shape[0]
        total_len = k_all.shape[0]
        past_len = total_len - seq_len
        causal_mask = torch.ones(seq_len, total_len, device=hidden.device, dtype=torch.bool).triu(diagonal=past_len + 1)
        attn_scores = torch.matmul(q, k_all.T) / math.sqrt(self.hidden_size)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_all)
        attn_output = self.out_proj(attn_output)

        # update cache with new kv
        self.kv_manager.append(user_tensor, self.layer_idx, k_new.unsqueeze(1), v_new.unsqueeze(1))

        hidden_out = hidden + self.dropout(attn_output)
        hidden_out = hidden_out + self.ffn(hidden_out)
        return hidden_out

    def forward_batch(self, hidden: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        # hidden shape: (seq_len, batch, hidden)
        outputs: List[torch.Tensor] = []
        for b_idx, uid in enumerate(user_ids.tolist()):
            single_hidden = hidden[:, b_idx, :]
            out = self.forward_single(single_hidden, int(uid))
            outputs.append(out.unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def forward(self, hidden: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
        if user_ids.dim() == 0:
            user_ids = user_ids.unsqueeze(0)
        return self.forward_batch(hidden, user_ids)


class HSTUModel(nn.Module):
    def __init__(self, model_config: ModelConfig, kv_config: KVCacheConfig, device: torch.device) -> None:
        super().__init__()
        self.device = device
        kv_layers = kv_config.num_layers or model_config.num_layers
        self.kv_manager = KVCacheManager(
            num_layers=kv_layers,
            hidden_size=model_config.hidden_size,
            max_length=kv_config.max_cache_length,
            device=device,
        )
        self.embedding = nn.Embedding(model_config.vocab_size, model_config.hidden_size)
        self.position = PositionalEncoding(model_config.hidden_size, model_config.max_position_embeddings, device)
        self.layers = nn.ModuleList(
            [HSTULayer(model_config, idx, self.kv_manager) for idx in range(model_config.num_layers)]
        )
        self.norm = nn.LayerNorm(model_config.hidden_size)
        self.head = nn.Linear(model_config.hidden_size, model_config.vocab_size)
        self.to(device)
        self.cast(model_config.dtype)

    def cast(self, dtype: torch.dtype) -> None:
        self.embedding = self.embedding.to(dtype=dtype)
        self.position = self.position.to(dtype=dtype)
        self.layers = self.layers.to(dtype=dtype)
        self.norm = self.norm.to(dtype=dtype)
        self.head = self.head.to(dtype=dtype)

    def forward(self, tokens: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        # tokens: (seq_len,) or (seq_len, batch)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(1)
        embeddings = self.embedding(tokens)
        embeddings = self.position(embeddings)
        hidden = embeddings
        for layer in self.layers:
            hidden = layer(hidden, user_ids)
        hidden = self.norm(hidden)
        logits = self.head(hidden)
        return logits

    def get_cache_info(self, user_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kv_manager.get_user_kvdata_info(user_ids)

    def clear_cache(self, user_ids: torch.Tensor | None = None) -> None:
        self.kv_manager.evict(user_ids)


def build_model(model_config: ModelConfig, kv_config: KVCacheConfig, device: str | None = None) -> HSTUModel:
    torch_device = _build_device(device)
    return HSTUModel(model_config, kv_config, torch_device)
