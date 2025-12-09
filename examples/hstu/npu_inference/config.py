"""Configuration objects for HSTU NPU inference.

The classes here are intentionally self contained so that the
NPU inference entrypoints do not depend on code outside this
folder.  Only standard library modules and :mod:`torch` are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


class InferenceConfig:
    """Top level configuration for running inference."""

    batch_size: int = 1
    sequence_length: int = 32
    device: Optional[str] = None
    model: ModelConfig = ModelConfig()
    kv_cache: KVCacheConfig = KVCacheConfig()
    dataset: DatasetConfig = DatasetConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
