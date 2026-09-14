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
`unseen_mask`. It may select an official subset of missing training cells; the
script rejects any selected sample whose class-domain cell was observed in
training. Evaluation labels are used only to compute recall after predictions.

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

For the paper protocol, pool all eligible cell-by-seed episodes before making
one selection:

```bash
python -m experiments.select_loco_multiseed \
  --training caches/seed42_training.npz caches/seed43_training.npz caches/seed44_training.npz \
  --validation caches/seed42_validation.npz caches/seed43_validation.npz caches/seed44_validation.npz \
  --output outputs/three_seed_lock.json
```

## Held-cell evaluation

```bash
python -m experiments.evaluate \
  --training caches/seed42_training.npz \
  --evaluation caches/seed42_test.npz \
  --output outputs/seed42_result.json
```

This evaluates the 2x2 choice of global/local prototypes for observed cells
and global/RAPC prototypes for missing cells. It reports per-class recall and
macro `BA_unseen`. Final RAPC uses global prototypes for observed cells;
`local_rapc` is an inference-only ablation.

## Matched-seed aggregation

```bash
python -m experiments.aggregate \
  outputs/seed42_result.json \
  outputs/seed43_result.json \
  outputs/seed44_result.json \
  --output outputs/three_seed_summary.json
```

Encoder checkpoints, third-party representations, and caches stay outside Git.
