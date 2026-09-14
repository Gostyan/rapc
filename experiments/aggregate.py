"""Aggregate matched standalone evaluation JSON files across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def aggregate(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"{path} is not a complete evaluation")
        global_ba = float(payload["methods"]["global_prototype"]["BA_unseen"])
        rapc_ba = float(payload["methods"]["rapc"]["BA_unseen"])
        rows.append(
            {
                "path": str(path),
                "global_prototype": global_ba,
                "rapc": rapc_ba,
                "paired_gain": rapc_ba - global_ba,
            }
        )
    if not rows:
        raise ValueError("at least one result is required")
    global_values = np.asarray([row["global_prototype"] for row in rows])
    rapc_values = np.asarray([row["rapc"] for row in rows])
    gains = rapc_values - global_values
    return {
        "schema_version": "rapc_matched_aggregate.v1",
        "run_count": len(rows),
        "global_prototype_mean": float(global_values.mean()),
        "global_prototype_std_population": float(global_values.std(ddof=0)),
        "rapc_mean": float(rapc_values.mean()),
        "rapc_std_population": float(rapc_values.std(ddof=0)),
        "paired_gain_mean": float(gains.mean()),
        "positive_runs": int((gains > 0).sum()),
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate([path.resolve() for path in args.results])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
