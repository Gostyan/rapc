# Standalone experiments

These scripts operate on frozen embedding caches, so the repository does not
need to contain encoder training code or audio datasets.

## Cache format

Training and validation `.npz` files contain:

```text
embeddings  float array [N, L]
labels      integer array [N], zero-based class IDs
domains     integer array [N], zero-based known-domain IDs
```

Evaluation caches use the same arrays and may additionally contain a boolean
`unseen_mask`. If supplied, the script verifies that it exactly matches cells
absent from the training cache. Evaluation labels are used only to compute
recall after predictions have been produced.

## Validation-only selection

```bash
python -m experiments.select_loco \
  --training caches/seed42_training.npz \
  --validation caches/seed42_validation.npz \
  --output outputs/seed42_lock.json
```

The script temporarily hides each eligible observed training cell and scores
the corresponding validation cell. Its interface deliberately has no test
cache argument.

## Held-cell evaluation

```bash
python -m experiments.evaluate \
  --training caches/seed42_training.npz \
  --evaluation caches/seed42_test.npz \
  --output outputs/seed42_result.json
```

This compares the Global prototype baseline with final locked ProtoFill and
reports per-class recall plus macro `BA_unseen`.

## Matched-seed aggregation

```bash
python -m experiments.aggregate \
  outputs/seed42_result.json \
  outputs/seed43_result.json \
  outputs/seed44_result.json \
  --output outputs/three_seed_summary.json
```

Encoder checkpoints, third-party representations, and caches stay outside Git.
