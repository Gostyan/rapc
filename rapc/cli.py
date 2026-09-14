"""Command-line interface for encoder-independent RAPC inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .core import DEFAULT_RIDGE_LAMBDA, DEFAULT_STRENGTH, build_domain_prototypes, fit_rapc, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit RAPC from an NPZ training cache and classify query embeddings. "
            "Required keys: train_embeddings, train_labels, train_domains, "
            "query_embeddings, query_domains."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--num-domains", type=int, default=None)
    parser.add_argument("--ridge-lambda", type=float, default=DEFAULT_RIDGE_LAMBDA)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as cache:
        required = {
            "train_embeddings",
            "train_labels",
            "train_domains",
            "query_embeddings",
            "query_domains",
        }
        missing = sorted(required.difference(cache.files))
        if missing:
            raise ValueError(f"input cache lacks required keys: {missing}")
        fitted = fit_rapc(
            torch.from_numpy(cache["train_embeddings"]),
            torch.from_numpy(cache["train_labels"]),
            torch.from_numpy(cache["train_domains"]),
            num_classes=args.num_classes,
            num_domains=args.num_domains,
            ridge_lambda=args.ridge_lambda,
        )
        predictions = predict(
            fitted,
            torch.from_numpy(cache["query_embeddings"]),
            torch.from_numpy(cache["query_domains"]),
            strength=args.strength,
        )

    domain_tables = []
    domain_roles = []
    for domain in range(fitted.table.num_domains):
        table, roles = build_domain_prototypes(fitted, domain, strength=args.strength)
        domain_tables.append(table.cpu().numpy())
        domain_roles.append(roles)
    metadata = {
        "method": "RAPC",
        "ridge_lambda": float(args.ridge_lambda),
        "strength": float(args.strength),
        "geometry": "direction_nlerp",
        "cell_weighting": "equal",
        "uses_query_labels": False,
        "uses_query_statistics": False,
        "design_rank": fitted.design_rank,
        "unique_solution": fitted.unique_solution,
        "roles_by_domain": domain_roles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        predictions=predictions.cpu().numpy(),
        domain_prototypes=np.stack(domain_tables),
        observed_cells=fitted.table.observed.cpu().numpy(),
        cell_counts=fitted.table.counts.cpu().numpy(),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


if __name__ == "__main__":
    main()
