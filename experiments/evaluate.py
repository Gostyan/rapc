"""Evaluate locked RAPC and its 2x2 prototype-table controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from rapc import fit_rapc, predict

from .common import held_cell_mask, infer_sizes, load_cache, macro_recall, sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "configs/final_rapc.json"


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
        raise ValueError(f"unexpected RAPC lock: {observed}")
    return lock


def evaluate(training_path: Path, evaluation_path: Path, lock_path: Path) -> dict:
    training = load_cache(training_path)
    evaluation = load_cache(evaluation_path, allow_unseen_mask=True)
    if training.embeddings.shape[1] != evaluation.embeddings.shape[1]:
        raise ValueError("training and evaluation embedding dimensions differ")
    lock = load_lock(lock_path)
    num_classes, num_domains = infer_sizes(training, evaluation)
    fitted = fit_rapc(
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
    rapc_predictions = predict(
        fitted,
        evaluation.embeddings,
        evaluation.domains,
        strength=float(lock["strength_g"]),
    )
    local_global_predictions = predict(
        fitted, evaluation.embeddings, evaluation.domains,
        strength=0.0, observed_policy="local",
    )
    local_rapc_predictions = predict(
        fitted, evaluation.embeddings, evaluation.domains,
        strength=float(lock["strength_g"]), observed_policy="local",
    )
    inferred_mask = held_cell_mask(
        fitted.table.counts, evaluation.labels, evaluation.domains
    )
    mask = evaluation.unseen_mask if evaluation.unseen_mask is not None else inferred_mask
    if evaluation.unseen_mask is not None and bool((mask & ~inferred_mask).any()):
        raise ValueError("unseen_mask includes a class-domain cell observed in training")
    if not bool(mask.any()):
        raise ValueError("evaluation mask selects no sample")

    global_result = macro_recall(
        global_predictions, evaluation.labels, mask, num_classes
    )
    rapc_result = macro_recall(
        rapc_predictions, evaluation.labels, mask, num_classes
    )
    local_global_result = macro_recall(
        local_global_predictions, evaluation.labels, mask, num_classes
    )
    local_rapc_result = macro_recall(
        local_rapc_predictions, evaluation.labels, mask, num_classes
    )
    return {
        "schema_version": "rapc_standalone_evaluation.v1",
        "status": "complete",
        "protocol": {
            "known_query_domain_required": True,
            "query_labels_used_for_prototypes": False,
            "query_statistics_used_for_prototypes": False,
            "evaluation_labels_used_for_metrics_only": True,
            "unseen_mask": (
                "all missing training cells by default; an explicit registered "
                "subset may be supplied by the evaluation cache"
            ),
            "observed_missing_source": "training cache only",
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
            "local_global_prototype": local_global_result,
            "rapc": rapc_result,
            "local_rapc": local_rapc_result,
        },
        "paired_gain": (
            rapc_result["BA_unseen"] - global_result["BA_unseen"]
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
