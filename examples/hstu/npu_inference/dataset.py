"""Dataset helpers for KuaiRand benchmark inference."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import pandas as pd
import torch

from .config import DatasetConfig
from .preprocess_kuairand import preprocess_kuairand


@dataclass
class Sample:
    user_id: int | torch.Tensor
    tokens: torch.Tensor  # shape: (seq_len,)
    target: int | torch.Tensor
    sequence_id: str | None = None


class KuaiRandDataset:
    """Iterate over preprocessed KuaiRand sessions for inference."""

    def __init__(self, cfg: DatasetConfig, sequence_length: int) -> None:
        self.cfg = cfg
        self.sequence_length = sequence_length
        paths = preprocess_kuairand(
            dataset_name=cfg.dataset_name,
            data_root=cfg.data_root,
            time_interval_s=cfg.time_interval_s,
            max_sequences=cfg.max_sequences,
            max_sequence_length=cfg.max_sequence_length,
        )
        self.sequence_file = paths.sequence_file
        self.metadata = json.loads(Path(paths.metadata_file).read_text())
        self._frame = pd.read_csv(self.sequence_file)

    @property
    def vocab_size(self) -> int:
        return int(self.metadata.get("vocab_size", 0))

    def __len__(self) -> int:
        return len(self._frame)

    def _prepare_sequence(self, raw_tokens: List[int]) -> torch.Tensor:
        if len(raw_tokens) <= 1:
            raise ValueError("Sequence must contain at least two tokens")
        context = raw_tokens[:-1]
        if len(context) > self.sequence_length:
            context = context[-self.sequence_length :]
        padding = [0] * max(0, self.sequence_length - len(context))
        context_tokens = torch.tensor(padding + context, dtype=torch.long)
        return context_tokens

    def __iter__(self) -> Iterator[Sample]:
        for idx, row in self._frame.iterrows():
            tokens = json.loads(row["sequence"])
            if len(tokens) < 2:
                continue
            context_tokens = self._prepare_sequence(tokens)
            target = int(tokens[-1])
            yield Sample(
                user_id=int(row["user_id"]),
                tokens=context_tokens,
                target=target,
                sequence_id=f"{row['user_id']}-{idx}",
            )


def collate_samples(samples: Sequence[Sample], device: torch.device) -> Sample:
    tokens = torch.stack([s.tokens for s in samples], dim=1).to(device)
    targets = torch.tensor([s.target for s in samples], dtype=torch.long, device=device)
    user_ids = torch.tensor([s.user_id for s in samples], dtype=torch.long, device=device)
    sequence_ids = [s.sequence_id for s in samples]
    return Sample(user_id=user_ids, tokens=tokens, target=targets, sequence_id=",".join(sequence_ids))
