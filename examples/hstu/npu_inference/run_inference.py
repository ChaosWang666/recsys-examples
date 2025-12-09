"""Entry point for running HSTU inference on NPU/CPU.

The script mirrors the structure of ``examples/hstu/inference`` but is
self contained and free from GPU specific dependencies.  A lightweight
synthetic dataset is used to demonstrate incremental KV cache behaviour.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch

from .config import KVCacheConfig, ModelConfig
from .dataset import SyntheticDataset, collate_samples
from .model import build_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HSTU inference on NPU/CPU")
    parser.add_argument("--steps", type=int, default=4, help="Number of batches to process")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--sequence_length", type=int, default=16, help="Number of tokens per request")
    parser.add_argument("--num_users", type=int, default=8, help="Number of synthetic users")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads per layer")
    parser.add_argument("--device", type=str, default=None, help="Target device (npu/cpu/cuda)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Computation dtype")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to a saved state dict")
    parser.add_argument("--dump_logits", action="store_true", help="Print logits for debugging")
    return parser.parse_args()


def _dtype_from_string(name: str) -> torch.dtype:
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


def main() -> None:
    args = _parse_args()
    dtype = _dtype_from_string(args.dtype)

    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dtype=dtype,
    )
    kv_config = KVCacheConfig(max_cache_length=args.sequence_length * 4, num_layers=args.num_layers)
    model = build_model(model_config, kv_config, device=args.device)
    if args.checkpoint:
        _load_checkpoint(model, args.checkpoint)
    model.eval()

    dataset = SyntheticDataset(
        num_users=args.num_users,
        vocab_size=args.vocab_size,
        sequence_length=args.sequence_length,
        steps=args.steps,
    )

    results: List[Dict[str, torch.Tensor]] = []
    iterator = iter(dataset)
    for _ in range(args.steps):
        batch_samples = [next(iterator) for _ in range(args.batch_size)]
        batch = collate_samples(batch_samples, device=model.device)
        user_ids = batch.user_id if isinstance(batch.user_id, torch.Tensor) else torch.tensor([batch.user_id], device=model.device)
        logits = model(batch.tokens, user_ids)
        start_pos, cached_len = model.get_cache_info(user_ids)
        results.append({"start_pos": start_pos.cpu(), "cached_len": cached_len.cpu()})
        if args.dump_logits:
            print(f"Logits shape: {tuple(logits.shape)}")

    summary = {
        "num_requests": len(results),
        "cache_lengths": [int(r["cached_len"][i]) for r in results for i in range(r["cached_len"].numel())],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
