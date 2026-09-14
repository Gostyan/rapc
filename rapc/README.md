# RAPC

RAPC is an encoder-agnostic, inference-time module for completing a class
prototype that is absent from a **known** domain. It operates only on frozen
training embeddings, class labels, and domain labels. It does not update the
encoder and does not use query labels or query-set statistics.

This directory is the compact public implementation of the final method. The
validation-locked defaults are:

- two-way ridge strength `lambda = 1`;
- normalized-direction interpolation strength `g = 0.75`;
- equal weight for every observed class-domain cell.

The older local-anchor translation with `g = 0.5` is an ablation baseline and
is **not** the final RAPC method.

## Problem setting

Given normalized encoder outputs, let `p[d,c]` be the mean embedding of an
observed domain-class cell. RAPC fits

```text
p[d,c] = common_center + class_effect[c] + domain_effect[d]
```

using equally weighted observed cells and ridge-regularized class/domain
effects. A missing center is reconstructed from the fitted effects and then
conservatively interpolated with the sample-weighted global class direction.
Observed entries keep the global class prototype. If the target class and
target domain are disconnected in the observed bipartite support graph, the
method also falls back to the global prototype.

This is missing-cell inference with a known query-domain identifier. It is not
a method for a wholly unseen domain.

## Python API

```python
from rapc import fit_rapc, predict

fitted = fit_rapc(
    train_embeddings,  # [N, L]
    train_class_ids,   # [N], zero-based
    train_domain_ids,  # [N], zero-based
)

predictions = predict(
    fitted,
    query_embeddings,
    query_domain_ids,
)
```

Input embeddings are normalized sample by sample inside `fit_rapc` and
`predict`. Supplying query class labels is neither required nor supported.

## Command-line API

Prepare one `.npz` file containing:

```text
train_embeddings, train_labels, train_domains,
query_embeddings, query_domains
```

Then run:

```bash
python -m rapc --input embeddings.npz --output predictions.npz
```

The output contains predictions, the completed prototype table for every
known domain, observed-cell masks, cell counts, and an auditable JSON metadata
record. Checkpoints and embedding caches are intentionally excluded from Git.

## Solution and complexity

For `K` observed cells, `C` classes, `D` domains, embedding dimension `L`, and
`P=1+C+D`, the ridge coefficients have the closed form

```text
Theta = (X^T X + lambda R)^(-1) X^T Y.
```

The objective is convex. With `lambda > 0`, the system is positive definite,
so the coefficients are the unique global optimum. The reference code uses a
pseudoinverse for numerical compatibility with the `lambda=0` ablation.

Dense preprocessing costs approximately
`O(K P^2 + K P L + P^3 + P^2 L)` and is performed once per training cache.
Online classification remains `C` cosine comparisons per query, i.e.
`O(C L)`, the same asymptotic cost as a global-prototype classifier.

## Reproducing the protocol

The standalone, encoder-independent experiment entry points are:

- validation-only leave-one-observed-cell-out selection:
  `experiments/select_loco.py`;
- held-cell RAPC and 2x2 observed/missing prototype-policy evaluation:
  `experiments/evaluate.py`;
- joint multi-seed LOCO selection: `experiments/select_loco_multiseed.py`;
- matched-seed aggregation: `experiments/aggregate.py`.

All parameter selection must finish using training/validation embeddings
before a test cache is opened. The compact package above deliberately contains
no dataset paths, checkpoint paths, team names, or test labels.
