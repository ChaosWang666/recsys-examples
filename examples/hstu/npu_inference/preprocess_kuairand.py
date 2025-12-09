"""Minimal KuaiRand preprocessing tailored for NPU inference.

The logic borrows the public preprocessing steps from ``examples/hstu``
but focuses solely on the inference path.  The processed files live
under ``npu_inference/tmp_data`` so the scripts in this folder remain
self contained.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.request import urlretrieve

import pandas as pd

_DOWNLOAD_URLS: Dict[str, str] = {
    "kuairand-1k": "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz",
}

_LOG_FILES: Dict[str, List[str]] = {
    "kuairand-1k": [
        "log_standard_4_08_to_4_21_1k.csv",
        "log_standard_4_22_to_5_08_1k.csv",
    ],
}

_EVENT_WEIGHTS: Dict[str, int] = {
    "is_click": 1,
    "is_like": 2,
    "is_follow": 4,
    "is_comment": 8,
    "is_forward": 16,
    "is_hate": 32,
    "long_view": 64,
    "is_profile_enter": 128,
}


@dataclass
class KuaiRandPaths:
    dataset_dir: Path
    sequence_file: Path
    metadata_file: Path


class DownloadProgress:
    def __init__(self) -> None:
        self._last_percent = -1

    def __call__(self, count: int, block_size: int, total_size: int) -> None:
        percent = int(count * block_size * 100 / max(total_size, 1))
        if percent != self._last_percent and percent % 10 == 0:
            self._last_percent = percent
            print(f"...{percent}% downloaded")


def _prepare_directories(data_root: str, dataset_name: str) -> KuaiRandPaths:
    dataset_dir = Path(data_root) / dataset_name
    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sequence_file = dataset_dir / "processed_inference_sequences.csv"
    metadata_file = dataset_dir / "metadata.json"
    return KuaiRandPaths(dataset_dir=dataset_dir, sequence_file=sequence_file, metadata_file=metadata_file)


def _download_if_needed(url: str, target: Path) -> None:
    if target.exists():
        return
    progress = DownloadProgress()
    print(f"Downloading {url} -> {target}")
    urlretrieve(url, target, reporthook=progress)


def _extract_if_needed(archive: Path, target_dir: Path) -> None:
    marker = target_dir / ".extracted"
    if marker.exists():
        return
    print(f"Extracting {archive} to {target_dir}")
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(path=target_dir)
    marker.touch()


def _build_action_weights(df: pd.DataFrame) -> pd.Series:
    weights = []
    for col, weight in _EVENT_WEIGHTS.items():
        if col not in df.columns:
            raise ValueError(f"Expected column {col} in KuaiRand logs")
        weights.append(df[col].apply(lambda seq: 0 if seq == 0 else weight))
    summed = pd.concat(weights, axis=1).sum(axis=1).astype(int)
    return summed


def _gather_sequences(df: pd.DataFrame, interval_ms: int) -> Iterable[Tuple[int, List[int], List[int]]]:
    df = df.sort_values(["user_id", "time_ms"]).reset_index(drop=True)
    for user_id, user_df in df.groupby("user_id"):
        last_ts = None
        current_videos: List[int] = []
        current_actions: List[int] = []
        for row in user_df.itertuples(index=False):
            if last_ts is not None and row.time_ms - last_ts >= interval_ms:
                if current_videos:
                    yield user_id, current_videos, current_actions
                current_videos = []
                current_actions = []
            current_videos.append(int(row.video_id))
            current_actions.append(int(row.action_weights))
            last_ts = row.time_ms
        if current_videos:
            yield user_id, current_videos, current_actions


def preprocess_kuairand(
    *,
    dataset_name: str,
    data_root: str,
    time_interval_s: int,
    max_sequences: int,
    max_sequence_length: int,
) -> KuaiRandPaths:
    if dataset_name not in _DOWNLOAD_URLS:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    paths = _prepare_directories(data_root, dataset_name)
    if paths.sequence_file.exists() and paths.metadata_file.exists():
        return paths

    archive_path = paths.dataset_dir / f"{dataset_name}.tar.gz"
    _download_if_needed(_DOWNLOAD_URLS[dataset_name], archive_path)
    _extract_if_needed(archive_path, paths.dataset_dir)

    base_data_dir = paths.dataset_dir / "data"
    csv_files = [base_data_dir / name for name in _LOG_FILES[dataset_name]]
    frames: List[pd.DataFrame] = []
    for file in csv_files:
        if not file.exists():
            raise FileNotFoundError(f"Expected KuaiRand log file {file}")
        df = pd.read_csv(file, delimiter=",")
        df["action_weights"] = _build_action_weights(df)
        frames.append(df[["user_id", "time_ms", "video_id", "action_weights"]])

    merged = pd.concat(frames, axis=0)
    interval_ms = time_interval_s * 1000

    rows = []
    vocab = 0
    for user_id, videos, actions in _gather_sequences(merged, interval_ms):
        if not videos:
            continue
        vocab = max(vocab, max(videos))
        if len(videos) <= 1:
            continue
        videos = videos[-(max_sequence_length + 1) :]
        actions = actions[-(max_sequence_length + 1) :]
        if len(rows) >= max_sequences:
            break
        rows.append(
            {
                "user_id": user_id,
                "sequence": json.dumps(videos),
                "actions": json.dumps(actions),
                "length": len(videos),
            }
        )

    if not rows:
        raise RuntimeError("No usable KuaiRand sessions found for inference")

    Path(paths.sequence_file).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths.sequence_file, index=False)
    metadata = {
        "dataset_name": dataset_name,
        "num_sequences": len(rows),
        "max_sequence_length": max_sequence_length,
        "time_interval_s": time_interval_s,
        "vocab_size": vocab + 1,
        "num_users": len({row["user_id"] for row in rows}),
    }
    with paths.metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess KuaiRand data for NPU inference")
    parser.add_argument("--dataset_name", type=str, default="kuairand-1k", help="Dataset preset to download")
    parser.add_argument("--data_root", type=str, default="tmp_data", help="Root directory to store processed data")
    parser.add_argument("--time_interval_s", type=int, default=300, help="Session boundary in seconds")
    parser.add_argument("--max_sequences", type=int, default=1024, help="Maximum number of sequences to keep")
    parser.add_argument("--max_sequence_length", type=int, default=256, help="Truncate sequences to this length")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = preprocess_kuairand(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        time_interval_s=args.time_interval_s,
        max_sequences=args.max_sequences,
        max_sequence_length=args.max_sequence_length,
    )
    print(f"Preprocessed sequences written to: {paths.sequence_file}")
    print(f"Metadata saved to: {paths.metadata_file}")


if __name__ == "__main__":
    main()
