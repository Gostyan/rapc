"""Select ProtoFill ridge and interpolation strengths by validation-only LOCO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from protofill import fit_protofill, predict

from .common import infer_sizes, load_cache, sha256


DEFAULT_LAMBDAS = (0.0, 0.1, 1.0, 10.0)
DEFAULT_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def select(
    training_path: Path,
    validation_path: Path,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    strengths: tuple[float, ...] = DEFAULT_STRENGTHS,
) -> dict:
    training = load_cache(training_path)
    validation = load_cache(validation_path)
    if training.embeddings.shape[1] != validation.embeddings.shape[1]:
        raise ValueError("training and validation embedding dimensions differ")
    num_classes, num_domains = infer_sizes(training, validation)
    episodes = []
    grid = {(ridge, g): [] for ridge in lambdas for g in strengths}
    validation_cells = sorted(
        set(zip(validation.domains.tolist(), validation.labels.tolist()))
    )
    for domain, class_index in validation_cells:
        query_mask = (validation.domains == domain) & (validation.labels == class_index)
        hidden_mask = (training.domains == domain) & (training.labels == class_index)
        if not bool(hidden_mask.any()):
            continue
        keep = ~hidden_mask
        if not bool((training.labels[keep] == class_index).any()):
            continue
        episode_scores = {}
        for ridge in lambdas:
            fitted = fit_protofill(
                training.embeddings[keep],
                training.labels[keep],
                training.domains[keep],
                num_classes=num_classes,
                num_domains=num_domains,
                ridge_lambda=float(ridge),
            )
            for g in strengths:
                predictions = predict(
                    fitted,
                    validation.embeddings[query_mask],
                    validation.domains[query_mask],
                    strength=float(g),
                )
                recall = float((predictions == class_index).float().mean())
                grid[(ridge, g)].append(recall)
                episode_scores[f"lambda={ridge},g={g}"] = recall
        episodes.append(
            {
                "hidden_domain": int(domain),
                "hidden_class": int(class_index),
                "validation_count": int(query_mask.sum()),
                "recall": episode_scores,
            }
        )
    if not episodes:
        raise ValueError("no eligible leave-one-observed-cell-out episode")

    summary = {}
    candidates = []
    for (ridge, g), values in grid.items():
        if len(values) != len(episodes):
            raise RuntimeError("incomplete LOCO grid")
        mean = float(np.mean(values))
        key = f"lambda={ridge},g={g}"
        summary[key] = {"cell_macro_recall": mean, "episode_count": len(values)}
        candidates.append((mean, ridge, g))
    mean, ridge, g = max(candidates, key=lambda row: (row[0], row[1], -row[2]))
    return {
        "schema_version": "protofill_loco_selection.v1",
        "status": "locked_before_test_evaluation",
        "protocol": {
            "selection_data": "training and validation embeddings only",
            "test_cache_read": False,
            "test_labels_read": False,
            "cell_weighting": "equal",
            "geometry": "direction_nlerp",
            "tie_break": "larger lambda, then smaller g",
        },
        "source": {
            "training_cache": str(training_path),
            "training_sha256": sha256(training_path),
            "validation_cache": str(validation_path),
            "validation_sha256": sha256(validation_path),
        },
        "selected": {
            "ridge_lambda": float(ridge),
            "strength_g": float(g),
            "cell_macro_recall": mean,
        },
        "grid": summary,
        "episodes": episodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--strengths", nargs="+", type=float, default=list(DEFAULT_STRENGTHS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = select(
        args.training.resolve(),
        args.validation.resolve(),
        tuple(args.lambdas),
        tuple(args.strengths),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
