"""Entry point for running HSTU inference on NPU/CPU."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .config import KVCacheConfig, ModelConfig, get_config
from .dataset import KuaiRandDataset, Sample, collate_samples
from .model import build_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HSTU inference on NPU/CPU")
    parser.add_argument(
        "--config",
        type=str,
        default="kuairand_gr",
        help="Name of the preset config inside npu_inference.config",
    )
    parser.add_argument("--steps", type=int, default=None, help="Number of batches to process")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference")
    parser.add_argument("--sequence_length", type=int, default=None, help="Number of tokens per request")
    parser.add_argument("--device", type=str, default=None, help="Target device (npu/cpu/cuda)")
    parser.add_argument("--dtype", type=str, default=None, help="Computation dtype override")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to a saved state dict")
    parser.add_argument("--dump_logits", action="store_true", help="Print logits for debugging")
    return parser.parse_args()


def _dtype_from_string(name: str | None, default: torch.dtype) -> torch.dtype:
    if name is None:
        return default
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _load_checkpoint(model, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)


def _update_config_with_args(config, args: argparse.Namespace):
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.sequence_length:
        config.sequence_length = args.sequence_length
    if args.device:
        config.device = args.device
    if args.dtype:
        config.model.dtype = _dtype_from_string(args.dtype, config.model.dtype)
    return config


def _resolve_checkpoint_path(config, args: argparse.Namespace) -> Tuple[str | None, bool]:
    path = args.checkpoint or config.model.checkpoint_path
    if path is None:
        return None, False
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint {ckpt_path} not found. Download the model before running inference."
        )
    return str(ckpt_path), True


def _benchmark_metrics(latencies: List[float], hits: List[int], count: int) -> Dict[str, float]:
    if count == 0:
        return {}
    avg_latency = sum(latencies) / count
    return {
        "requests": count,
        "avg_latency_ms": avg_latency * 1000,
        "p50_latency_ms": sorted(latencies)[int(0.5 * count)] * 1000,
        "hit_rate": sum(hits) / max(1, len(hits)),
        "throughput_rps": count / sum(latencies),
    }


def _summarize_topk(logits: torch.Tensor, targets: torch.Tensor, top_k: int) -> Dict[str, List[int]]:
    last_logits = logits[-1]
    values, indices = torch.topk(last_logits, k=top_k, dim=-1)
    hits = []
    summaries: List[Dict[str, int]] = []
    for b in range(indices.shape[0]):
        topk_tokens = indices[b].tolist()
        target_token = int(targets[b])
        hits.append(1 if target_token in topk_tokens else 0)
        summaries.append({"target": target_token, "topk": topk_tokens})
    return {"hits": hits, "summaries": summaries}


def _run_batch(model, batch: Sample, top_k: int, dump_logits: bool):
    logits = model(batch.tokens, batch.user_id)
    if dump_logits:
        print(f"Logits shape: {tuple(logits.shape)}")
    metrics = _summarize_topk(logits, batch.target, top_k)
    return logits, metrics


def main() -> None:
    args = _parse_args()
    config = get_config(args.config)
    config = _update_config_with_args(config, args)
    dtype = config.model.dtype

    dataset = KuaiRandDataset(config.dataset, config.sequence_length)
    vocab_size = dataset.vocab_size or config.model.vocab_size
    model_config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        max_position_embeddings=config.model.max_position_embeddings,
        dtype=dtype,
    )
    kv_config = KVCacheConfig(
        max_cache_length=config.kv_cache.max_cache_length,
        num_layers=config.kv_cache.num_layers,
    )
    model = build_model(model_config, kv_config, device=config.device)
    checkpoint_path, model_downloaded = _resolve_checkpoint_path(config, args)
    if checkpoint_path:
        _load_checkpoint(model, checkpoint_path)
    else:
        print("No checkpoint provided; using randomly initialized weights.")
    model.eval()

    steps = args.steps or len(dataset)
    iterator = iter(dataset)
    results: List[Dict[str, torch.Tensor]] = []
    latencies: List[float] = []
    hits: List[int] = []
    batch_size = config.batch_size
    processed_batches = 0

    start_time = time.perf_counter()
    for _ in range(steps):
        batch_samples = []
        try:
            for _ in range(batch_size):
                batch_samples.append(next(iterator))
        except StopIteration:
            break
        batch = collate_samples(batch_samples, device=model.device)
        t0 = time.perf_counter()
        logits, metrics = _run_batch(model, batch, config.benchmark.top_k, args.dump_logits)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        hits.extend(metrics["hits"])
        start_pos, cached_len = model.get_cache_info(batch.user_id)
        results.append({"start_pos": start_pos.cpu(), "cached_len": cached_len.cpu(), "topk": metrics["summaries"]})
        processed_batches += 1

    total_time = time.perf_counter() - start_time
    summary = {
        "config": args.config,
        "dataset": config.dataset.dataset_name,
        "num_batches": processed_batches,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "model_checkpoint": checkpoint_path,
        "model_downloaded": model_downloaded,
        "total_time_s": total_time,
    }
    summary.update(_benchmark_metrics(latencies, hits, processed_batches))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
