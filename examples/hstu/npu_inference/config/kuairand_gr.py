"""KuaiRand GR ranking preset for NPU inference."""
from __future__ import annotations

import torch

from ..config import BenchmarkConfig, DatasetConfig, InferenceConfig, KVCacheConfig, ModelConfig

KUAI_RAND_GR_CONFIG = InferenceConfig()
KUAI_RAND_GR_CONFIG.batch_size = 2
KUAI_RAND_GR_CONFIG.sequence_length = 128
KUAI_RAND_GR_CONFIG.device = None
KUAI_RAND_GR_CONFIG.model = ModelConfig(
    vocab_size=120_000,
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    dropout=0.05,
    max_position_embeddings=4096,
    dtype=torch.bfloat16,
)
KUAI_RAND_GR_CONFIG.kv_cache = KVCacheConfig(max_cache_length=4096, num_layers=6)
KUAI_RAND_GR_CONFIG.dataset = DatasetConfig(
    dataset_name="kuairand-1k",
    data_root="tmp_data",
    time_interval_s=300,
    max_sequences=1024,
    max_sequence_length=256,
)
KUAI_RAND_GR_CONFIG.benchmark = BenchmarkConfig(top_k=5, warmup_batches=1)
