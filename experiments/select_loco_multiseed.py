"""Select one RAPC configuration jointly over matched encoder seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .select_loco import DEFAULT_LAMBDAS, DEFAULT_STRENGTHS, select


def select_multiseed(
    training_paths: list[Path],
    validation_paths: list[Path],
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    strengths: tuple[float, ...] = DEFAULT_STRENGTHS,
) -> dict:
    """Pool all eligible cell-by-seed LOCO episodes before selecting."""

    if not training_paths or len(training_paths) != len(validation_paths):
        raise ValueError("training and validation cache lists must have equal nonzero length")
    runs = [
        select(training, validation, lambdas=lambdas, strengths=strengths)
        for training, validation in zip(training_paths, validation_paths, strict=True)
    ]
    grid: dict[str, dict] = {}
    candidates = []
    for key in runs[0]["grid"]:
        values = []
        for run in runs:
            for episode in run["episodes"]:
                values.append(float(episode["recall"][key]))
        mean = float(np.mean(values))
        ridge_text, strength_text = key.split(",")
        ridge = float(ridge_text.split("=")[1])
        strength = float(strength_text.split("=")[1])
        grid[key] = {"cell_seed_macro_recall": mean, "episode_count": len(values)}
        candidates.append((mean, ridge, strength))
    mean, ridge, strength = max(
        candidates, key=lambda row: (row[0], row[1], -row[2])
    )
    return {
        "schema_version": "rapc_multiseed_loco_selection.v1",
        "status": "locked_before_test_evaluation",
        "protocol": {
            "selection_data": "training and validation embeddings only",
            "pooling": "equal weight per eligible cell-by-seed episode",
            "test_cache_read": False,
            "test_labels_read": False,
            "geometry": "direction_nlerp",
            "tie_break": "larger lambda, then smaller g",
        },
        "selected": {
            "ridge_lambda": ridge,
            "strength_g": strength,
            "cell_seed_macro_recall": mean,
        },
        "grid": grid,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", nargs="+", type=Path, required=True)
    parser.add_argument("--validation", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--strengths", nargs="+", type=float, default=list(DEFAULT_STRENGTHS))
    args = parser.parse_args()
    result = select_multiseed(
        [path.resolve() for path in args.training],
        [path.resolve() for path in args.validation],
        tuple(args.lambdas),
        tuple(args.strengths),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
