"""Minimal synthetic dataset helpers for NPU inference.

The original project contains rich preprocessing logic tied to external
components.  For NPU inference we only need a predictable way to generate
input token sequences so this module provides a reproducible synthetic
dataset that can be used to validate the model and KV cache logic.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List, Sequence

import torch


@dataclass
class Sample:
    user_id: int
    tokens: torch.Tensor  # shape: (seq_len,)


class SyntheticDataset:
    """Yield deterministic token sequences for a fixed set of users."""

    def __init__(
        self,
        num_users: int,
        vocab_size: int,
        sequence_length: int,
        steps: int,
        seed: int = 0,
    ) -> None:
        self.num_users = num_users
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.steps = steps
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[Sample]:
        for _ in range(self.steps):
            user_id = self._rng.randrange(self.num_users)
            tokens = [self._rng.randrange(self.vocab_size) for _ in range(self.sequence_length)]
            yield Sample(user_id=user_id, tokens=torch.tensor(tokens, dtype=torch.long))


def collate_samples(samples: Sequence[Sample], device: torch.device) -> Sample:
    """Merge a list of :class:`Sample` objects into a batch friendly structure."""
    if len(samples) == 1:
        sample = samples[0]
        return Sample(user_id=sample.user_id, tokens=sample.tokens.to(device))
    user_ids: List[int] = []
    token_tensors: List[torch.Tensor] = []
    for sample in samples:
        user_ids.append(sample.user_id)
        token_tensors.append(sample.tokens)
    stacked_tokens = torch.stack(token_tensors, dim=1).to(device)
    stacked_users = torch.tensor(user_ids, device=device, dtype=torch.long)
    return Sample(user_id=stacked_users, tokens=stacked_tokens)
