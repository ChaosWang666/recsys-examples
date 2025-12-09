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
- A lightweight synthetic dataset in [`dataset.py`](dataset.py)
  replaces the training-time preprocessing pipeline so the inference
  entrypoint can run without additional dependencies.

## Quick start

```bash
python -m examples.hstu.npu_inference.run_inference \
  --steps 2 \
  --batch_size 1 \
  --sequence_length 16 \
  --device npu
```

Use `--dump_logits` to print output tensor shapes for debugging or
`--checkpoint` to load a saved PyTorch state dict.
