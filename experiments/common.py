"""Shared cache loading and metrics for standalone ProtoFill experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class EmbeddingCache:
    embeddings: torch.Tensor
    labels: torch.Tensor
    domains: torch.Tensor
    unseen_mask: torch.Tensor | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path: Path, *, allow_unseen_mask: bool = False) -> EmbeddingCache:
    with np.load(path, allow_pickle=False) as data:
        required = {"embeddings", "labels", "domains"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path} lacks required arrays: {missing}")
        embeddings = torch.from_numpy(np.asarray(data["embeddings"])).float()
        labels = torch.from_numpy(np.asarray(data["labels"])).long()
        domains = torch.from_numpy(np.asarray(data["domains"])).long()
        unseen = None
        if allow_unseen_mask and "unseen_mask" in data.files:
            unseen = torch.from_numpy(np.asarray(data["unseen_mask"])).bool()
    if embeddings.ndim != 2:
        raise ValueError(f"{path}: embeddings must have shape [N, L]")
    if labels.ndim != 1 or domains.ndim != 1:
        raise ValueError(f"{path}: labels and domains must be one-dimensional")
    if not (len(embeddings) == len(labels) == len(domains)):
        raise ValueError(f"{path}: cache arrays have inconsistent lengths")
    if unseen is not None and (unseen.ndim != 1 or len(unseen) != len(labels)):
        raise ValueError(f"{path}: unseen_mask must have shape [N]")
    if len(labels) == 0 or int(labels.min()) < 0 or int(domains.min()) < 0:
        raise ValueError(f"{path}: cache must contain non-negative IDs and samples")
    return EmbeddingCache(embeddings, labels, domains, unseen)


def infer_sizes(training: EmbeddingCache, evaluation: EmbeddingCache) -> tuple[int, int]:
    unseen_classes = set(evaluation.labels.tolist()).difference(training.labels.tolist())
    unseen_domains = set(evaluation.domains.tolist()).difference(training.domains.tolist())
    if unseen_classes:
        raise ValueError(f"evaluation contains classes absent from training: {sorted(unseen_classes)}")
    if unseen_domains:
        raise ValueError(f"evaluation contains wholly unseen domains: {sorted(unseen_domains)}")
    num_classes = max(int(training.labels.max()), int(evaluation.labels.max())) + 1
    num_domains = max(int(training.domains.max()), int(evaluation.domains.max())) + 1
    return num_classes, num_domains


def held_cell_mask(
    training_counts: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
) -> torch.Tensor:
    return training_counts[domains, labels] == 0


def macro_recall(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> dict:
    per_class = {}
    recalls = []
    for class_index in range(num_classes):
        class_mask = mask & (labels == class_index)
        count = int(class_mask.sum())
        if not count:
            continue
        recall = float((predictions[class_mask] == class_index).float().mean())
        recalls.append(recall)
        per_class[str(class_index)] = {"recall": recall, "count": count}
    if not recalls:
        raise ValueError("evaluation mask contains no class")
    return {
        "BA_unseen": float(np.mean(recalls)),
        "num_macro_classes": len(recalls),
        "num_samples": int(mask.sum()),
        "per_class": per_class,
    }
