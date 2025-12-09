"""Configuration objects and presets for HSTU NPU inference."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class ModelConfig:
    """Configuration for the lightweight HSTU inference model."""

    vocab_size: int = 50_000
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.0
    max_position_embeddings: int = 1024
    dtype: torch.dtype = torch.bfloat16


@dataclass
class KVCacheConfig:
    """Configuration that governs KV cache behaviour."""

    max_cache_length: int = 2048
    num_layers: Optional[int] = None


@dataclass
class DatasetConfig:
    """Configuration for KuaiRand benchmark preparation and loading."""

    dataset_name: str = "kuairand-1k"
    data_root: str = "tmp_data"
    time_interval_s: int = 300
    max_sequences: int = 512
    max_sequence_length: int = 256


@dataclass
class BenchmarkConfig:
    """Options that control benchmark style output for ranking."""

    top_k: int = 5
    warmup_batches: int = 1


@dataclass
class InferenceConfig:
    """Top level configuration for running inference."""

    batch_size: int = 1
    sequence_length: int = 32
    device: Optional[str] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


KUAI_RAND_GR_CONFIG = InferenceConfig(
    batch_size=2,
    sequence_length=128,
    device=None,
    model=ModelConfig(
        vocab_size=120_000,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        dropout=0.05,
        max_position_embeddings=4096,
        dtype=torch.bfloat16,
    ),
    kv_cache=KVCacheConfig(max_cache_length=4096, num_layers=6),
    dataset=DatasetConfig(
        dataset_name="kuairand-1k",
        data_root="tmp_data",
        time_interval_s=300,
        max_sequences=1024,
        max_sequence_length=256,
    ),
    benchmark=BenchmarkConfig(top_k=5, warmup_batches=1),
)


_PRESETS: Dict[str, InferenceConfig] = {"kuairand_gr": KUAI_RAND_GR_CONFIG}


def get_config(name: str) -> InferenceConfig:
    """Return a deep copied preset configuration by name."""

    if name not in _PRESETS:
        raise ValueError(f"Unknown config preset: {name}")
    return copy.deepcopy(_PRESETS[name])


__all__ = [
    "BenchmarkConfig",
    "DatasetConfig",
    "InferenceConfig",
    "KVCacheConfig",
    "KUAI_RAND_GR_CONFIG",
    "ModelConfig",
    "get_config",
]
