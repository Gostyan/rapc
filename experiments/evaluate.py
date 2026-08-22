"""Evaluate Global prototype and locked ProtoFill on held class-domain cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from protofill import fit_protofill, predict

from .common import held_cell_mask, infer_sizes, load_cache, macro_recall, sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "protofill/artifacts/final_config.json"


def load_lock(path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "ridge_lambda": 1.0,
        "strength_g": 0.75,
        "geometry": "direction_nlerp",
        "cell_weighting": "equal",
    }
    observed = {key: lock.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"unexpected ProtoFill lock: {observed}")
    return lock


def evaluate(training_path: Path, evaluation_path: Path, lock_path: Path) -> dict:
    training = load_cache(training_path)
    evaluation = load_cache(evaluation_path, allow_unseen_mask=True)
    if training.embeddings.shape[1] != evaluation.embeddings.shape[1]:
        raise ValueError("training and evaluation embedding dimensions differ")
    lock = load_lock(lock_path)
    num_classes, num_domains = infer_sizes(training, evaluation)
    fitted = fit_protofill(
        training.embeddings,
        training.labels,
        training.domains,
        num_classes=num_classes,
        num_domains=num_domains,
        ridge_lambda=float(lock["ridge_lambda"]),
    )

    queries = F.normalize(evaluation.embeddings, dim=1)
    global_directions = F.normalize(fitted.table.global_centers, dim=1)
    global_predictions = (queries @ global_directions.T).argmax(1)
    protofill_predictions = predict(
        fitted,
        evaluation.embeddings,
        evaluation.domains,
        strength=float(lock["strength_g"]),
    )
    inferred_mask = held_cell_mask(
        fitted.table.counts, evaluation.labels, evaluation.domains
    )
    mask = evaluation.unseen_mask if evaluation.unseen_mask is not None else inferred_mask
    if evaluation.unseen_mask is not None and not torch.equal(mask, inferred_mask):
        raise ValueError("provided unseen_mask disagrees with missing training cells")

    global_result = macro_recall(
        global_predictions, evaluation.labels, mask, num_classes
    )
    protofill_result = macro_recall(
        protofill_predictions, evaluation.labels, mask, num_classes
    )
    return {
        "schema_version": "protofill_standalone_evaluation.v1",
        "status": "complete",
        "protocol": {
            "known_query_domain_required": True,
            "query_labels_used_for_prototypes": False,
            "query_statistics_used_for_prototypes": False,
            "evaluation_labels_used_for_metrics_only": True,
            "unseen_mask": "training-cell absence, optionally asserted by cache",
        },
        "configuration": lock,
        "source": {
            "training_cache": str(training_path),
            "training_sha256": sha256(training_path),
            "evaluation_cache": str(evaluation_path),
            "evaluation_sha256": sha256(evaluation_path),
        },
        "methods": {
            "global_prototype": global_result,
            "protofill": protofill_result,
        },
        "paired_gain": (
            protofill_result["BA_unseen"] - global_result["BA_unseen"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args.training.resolve(), args.evaluation.resolve(), args.lock.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "methods": result["methods"], "paired_gain": result["paired_gain"]}, indent=2))


if __name__ == "__main__":
    main()
