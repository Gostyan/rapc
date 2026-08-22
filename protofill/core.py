"""Closed-form completion of missing class-domain prototypes.

ProtoFill consumes frozen training embeddings, class labels, and known domain
labels.  It never uses query labels or query statistics to construct its
prototype table.  The public defaults are the validation-locked paper
configuration: ridge lambda=1, interpolation strength g=0.75, and normalized
direction interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_RIDGE_LAMBDA = 1.0
DEFAULT_STRENGTH = 0.75


@dataclass(frozen=True)
class CellTable:
    """Training-only sufficient statistics for a class-domain table."""

    centers: torch.Tensor
    counts: torch.Tensor
    global_centers: torch.Tensor

    @property
    def num_domains(self) -> int:
        return int(self.centers.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.centers.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.centers.shape[2])

    @property
    def observed(self) -> torch.Tensor:
        return self.counts > 0


@dataclass(frozen=True)
class ProtoFillFit:
    """A fitted two-way ridge model and the statistics that produced it."""

    table: CellTable
    coefficients: torch.Tensor
    ridge_lambda: float
    design_rank: int
    unique_solution: bool

    @property
    def num_parameters(self) -> int:
        return 1 + self.table.num_classes + self.table.num_domains


def _validate_ids(ids: torch.Tensor, name: str, size: int, rows: int) -> None:
    if ids.ndim != 1 or len(ids) != rows:
        raise ValueError(f"{name} must be a one-dimensional tensor of length {rows}")
    if len(ids) and (int(ids.min()) < 0 or int(ids.max()) >= size):
        raise ValueError(f"{name} must contain zero-based IDs in [0, {size})")


@torch.no_grad()
def build_cell_table(
    embeddings: torch.Tensor,
    class_ids: torch.Tensor,
    domain_ids: torch.Tensor,
    *,
    num_classes: int | None = None,
    num_domains: int | None = None,
    normalize_embeddings: bool = True,
    eps: float = 1e-12,
) -> CellTable:
    """Compute sample-weighted global centers and equally weighted cell centers.

    Embeddings are L2-normalized sample by sample by default, matching the
    paper protocol.  Empty classes are rejected because neither the baseline
    nor ProtoFill can construct their global class prototype.
    """

    embeddings = torch.as_tensor(embeddings).detach()
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("embeddings must be a non-empty [N, L] tensor")
    if not torch.is_floating_point(embeddings):
        embeddings = embeddings.float()
    class_ids = torch.as_tensor(class_ids, device=embeddings.device).long()
    domain_ids = torch.as_tensor(domain_ids, device=embeddings.device).long()
    inferred_classes = int(class_ids.max()) + 1 if len(class_ids) else 0
    inferred_domains = int(domain_ids.max()) + 1 if len(domain_ids) else 0
    num_classes = inferred_classes if num_classes is None else int(num_classes)
    num_domains = inferred_domains if num_domains is None else int(num_domains)
    if num_classes <= 0 or num_domains <= 0:
        raise ValueError("num_classes and num_domains must be positive")
    _validate_ids(class_ids, "class_ids", num_classes, len(embeddings))
    _validate_ids(domain_ids, "domain_ids", num_domains, len(embeddings))

    z = F.normalize(embeddings, dim=1, eps=eps) if normalize_embeddings else embeddings
    centers = z.new_zeros((num_domains, num_classes, z.shape[1]))
    counts = torch.zeros(
        (num_domains, num_classes), dtype=torch.long, device=z.device
    )
    for domain in range(num_domains):
        for class_index in range(num_classes):
            mask = (domain_ids == domain) & (class_ids == class_index)
            count = int(mask.sum())
            counts[domain, class_index] = count
            if count:
                centers[domain, class_index] = z[mask].mean(0)

    global_centers = z.new_empty((num_classes, z.shape[1]))
    for class_index in range(num_classes):
        mask = class_ids == class_index
        if not bool(mask.any()):
            raise ValueError(f"class {class_index} has no training sample")
        global_centers[class_index] = z[mask].mean(0)
    return CellTable(centers=centers, counts=counts, global_centers=global_centers)


def support_connected(
    observed: torch.Tensor,
    target_domain: int,
    target_class: int,
) -> bool:
    """Test connectivity in the observed class-domain bipartite graph."""

    if observed.ndim != 2:
        raise ValueError("observed must have shape [D, C]")
    num_domains, num_classes = map(int, observed.shape)
    if not 0 <= target_domain < num_domains or not 0 <= target_class < num_classes:
        raise ValueError("target IDs are outside the cell table")
    observed_cpu = observed.bool().cpu()
    start = ("class", int(target_class))
    goal = ("domain", int(target_domain))
    queue = [start]
    seen = {start}
    while queue:
        kind, index = queue.pop(0)
        if (kind, index) == goal:
            return True
        if kind == "class":
            neighbors = [
                ("domain", domain)
                for domain in range(num_domains)
                if bool(observed_cpu[domain, index])
            ]
        else:
            neighbors = [
                ("class", class_index)
                for class_index in range(num_classes)
                if bool(observed_cpu[index, class_index])
            ]
        for neighbor in neighbors:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return False


@torch.no_grad()
def fit_two_way_ridge(
    observed_centers: torch.Tensor,
    class_ids: torch.Tensor,
    domain_ids: torch.Tensor,
    *,
    num_classes: int,
    num_domains: int,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit center = intercept + class effect + domain effect.

    Every input row receives equal weight.  Coordinates are solved jointly as
    multiple right-hand sides of the same small ridge system.  A pseudoinverse
    is used to preserve the reference implementation at lambda=0; for
    lambda>0 the mathematical solution is unique.
    """

    observed_centers = torch.as_tensor(observed_centers).detach()
    if observed_centers.ndim != 2 or len(observed_centers) == 0:
        raise ValueError("observed_centers must be a non-empty [K, L] tensor")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be non-negative")
    device = observed_centers.device
    class_ids = torch.as_tensor(class_ids, device=device).long()
    domain_ids = torch.as_tensor(domain_ids, device=device).long()
    _validate_ids(class_ids, "class_ids", num_classes, len(observed_centers))
    _validate_ids(domain_ids, "domain_ids", num_domains, len(observed_centers))

    columns = 1 + int(num_classes) + int(num_domains)
    design = torch.zeros((len(observed_centers), columns), dtype=torch.float64, device=device)
    rows = torch.arange(len(observed_centers), device=device)
    design[:, 0] = 1.0
    design[rows, 1 + class_ids] = 1.0
    design[rows, 1 + int(num_classes) + domain_ids] = 1.0
    targets = observed_centers.to(dtype=torch.float64)
    penalty = torch.eye(columns, dtype=torch.float64, device=device) * float(ridge_lambda)
    penalty[0, 0] = 0.0
    normal = design.T @ design + penalty
    coefficients = torch.linalg.pinv(normal) @ design.T @ targets
    metadata = {
        "observed_cell_count": int(len(observed_centers)),
        "num_parameters": columns,
        "design_rank": int(torch.linalg.matrix_rank(design)),
        "ridge_lambda": float(ridge_lambda),
        "cell_weighting": "equal",
        "closed_form": True,
        "global_optimum": True,
        "unique_solution": bool(ridge_lambda > 0),
    }
    return coefficients.to(dtype=observed_centers.dtype), metadata


@torch.no_grad()
def fit_protofill(
    embeddings: torch.Tensor,
    class_ids: torch.Tensor,
    domain_ids: torch.Tensor,
    *,
    num_classes: int | None = None,
    num_domains: int | None = None,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    normalize_embeddings: bool = True,
) -> ProtoFillFit:
    """Fit ProtoFill from training data only."""

    table = build_cell_table(
        embeddings,
        class_ids,
        domain_ids,
        num_classes=num_classes,
        num_domains=num_domains,
        normalize_embeddings=normalize_embeddings,
    )
    indices = table.observed.nonzero(as_tuple=False)
    centers = table.centers[indices[:, 0], indices[:, 1]]
    coefficients, metadata = fit_two_way_ridge(
        centers,
        indices[:, 1],
        indices[:, 0],
        num_classes=table.num_classes,
        num_domains=table.num_domains,
        ridge_lambda=ridge_lambda,
    )
    return ProtoFillFit(
        table=table,
        coefficients=coefficients,
        ridge_lambda=float(ridge_lambda),
        design_rank=int(metadata["design_rank"]),
        unique_solution=bool(metadata["unique_solution"]),
    )


def _completed_center(
    fitted: ProtoFillFit,
    target_domain: int,
    target_class: int,
) -> torch.Tensor | None:
    if not support_connected(fitted.table.observed, target_domain, target_class):
        return None
    query = fitted.coefficients.new_zeros(fitted.num_parameters)
    query[0] = 1.0
    query[1 + target_class] = 1.0
    query[1 + fitted.table.num_classes + target_domain] = 1.0
    return query @ fitted.coefficients


def _unit(vector: torch.Tensor, eps: float) -> torch.Tensor | None:
    norm = vector.norm()
    if not bool(torch.isfinite(norm)) or float(norm) <= eps:
        return None
    return vector / norm


@torch.no_grad()
def build_domain_prototypes(
    fitted: ProtoFillFit,
    target_domain: int,
    *,
    strength: float = DEFAULT_STRENGTH,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Build the C-way prototype table for one known query domain.

    Observed entries retain the global class direction.  Only missing entries
    are replaced, and disconnected or numerically degenerate completions fall
    back to that same global direction.
    """

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    if not 0 <= target_domain < fitted.table.num_domains:
        raise ValueError("target_domain is outside the fitted table")
    global_directions = F.normalize(fitted.table.global_centers, dim=1, eps=eps)
    prototypes = global_directions.clone()
    details: list[dict[str, Any]] = []
    for class_index in range(fitted.table.num_classes):
        if bool(fitted.table.observed[target_domain, class_index]):
            details.append({"role": "observed_global_anchor"})
            continue
        completion = _completed_center(fitted, target_domain, class_index)
        completed_direction = None if completion is None else _unit(completion, eps)
        if completed_direction is None:
            details.append({"role": "missing_global_fallback"})
            continue
        chord = (1.0 - strength) * global_directions[class_index] + strength * completed_direction
        final_direction = _unit(chord, eps)
        if final_direction is None:
            details.append({"role": "missing_global_fallback"})
            continue
        prototypes[class_index] = final_direction
        details.append(
            {
                "role": "missing_completed",
                "ridge_lambda": fitted.ridge_lambda,
                "strength": float(strength),
                "geometry": "direction_nlerp",
                "connected_support_graph": True,
            }
        )
    return prototypes, details


@torch.no_grad()
def predict(
    fitted: ProtoFillFit,
    query_embeddings: torch.Tensor,
    query_domain_ids: torch.Tensor,
    *,
    strength: float = DEFAULT_STRENGTH,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Classify queries without using query labels or query-set statistics."""

    queries = torch.as_tensor(query_embeddings, device=fitted.coefficients.device)
    if queries.ndim != 2 or queries.shape[1] != fitted.table.dimension:
        raise ValueError("query_embeddings must have shape [Q, L]")
    query_domain_ids = torch.as_tensor(query_domain_ids, device=queries.device).long()
    _validate_ids(
        query_domain_ids,
        "query_domain_ids",
        fitted.table.num_domains,
        len(queries),
    )
    queries = F.normalize(queries.float(), dim=1, eps=eps)
    predictions = torch.empty(len(queries), dtype=torch.long, device=queries.device)
    for domain in query_domain_ids.unique(sorted=True).tolist():
        domain = int(domain)
        prototypes, _ = build_domain_prototypes(
            fitted, domain, strength=strength, eps=eps
        )
        mask = query_domain_ids == domain
        predictions[mask] = (queries[mask] @ prototypes.to(queries.dtype).T).argmax(1)
    return predictions
