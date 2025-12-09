# HSTU NPU Inference

This folder provides a minimal, self contained implementation of HSTU
inference that runs on NPU or CPU devices without depending on any
GPU-only libraries.  All code lives inside this directory so it can be
used independently from the rest of the repository.

## Key changes vs. the CUDA path
- KV cache management is rewritten with plain Torch operators in
  [`kv_cache.py`](kv_cache.py), removing the reliance on
  `tensorrt_llm`, `flashinfer`, `triton`, `dynamicemb`, and
  `paged_kvcache_ops`.
- The inference model in [`model.py`](model.py) uses standard
  `torch.nn` layers and positional encodings that work on NPU/CPU.
- [`preprocess_kuairand.py`](preprocess_kuairand.py) downloads and
  prepares the KuaiRand benchmark dataset into
  `npu_inference/tmp_data`, mirroring the higher level preprocessing
  pipeline without external dependencies.
- [`dataset.py`](dataset.py) loads the processed KuaiRand sessions and
  exposes batches that include user ids, token sequences, and ranking
  targets so the inference path can report benchmark-style metrics.
- Preset hyperparameters inspired by `examples/hstu/configs` live in
  [`config.py`](config.py), keeping runtime configuration local to this
  directory.

## Quick start

1. Preprocess KuaiRand-1K once (downloads to `tmp_data/KuaiRand-1K`):

```bash
python -m examples.hstu.npu_inference.preprocess_kuairand \
  --dataset_name "KuaiRand-1K" \
  --data_root tmp_data
```

2. Run inference on the generated benchmark data:

```bash
python -m examples.hstu.npu_inference.run_inference \
  --config kuairand_gr \
  --steps 4 \
  --device npu \
  --checkpoint /path/to/downloaded/model.pt
```

Use `--dump_logits` to print output tensor shapes for debugging or
`--checkpoint` to load a saved PyTorch state dict. If no checkpoint is
provided the script reports that random weights are used so you can
verify model downloads before benchmarking.

## Benchmark output

The entrypoint emits a JSON summary that includes:
- number of processed batches and batch size
- vocabulary size determined from the benchmark data
- latency statistics (average and p50) and throughput
- a simple hit-rate@K computed from the KuaiRand next-item targets

This mirrors the ranking oriented reporting style used by the CUDA
`inference_gr_ranking.py` script while staying compatible with NPU
execution.
