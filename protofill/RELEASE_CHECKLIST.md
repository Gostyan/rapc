# ProtoFill release checklist

## Included in the compact release

- encoder-independent PyTorch API in `protofill/core.py`;
- `.npz` command-line inference in `protofill/cli.py`;
- validation-locked final configuration;
- compact matched-run result summary with source locations;
- unit tests for the closed-form solution, graph fallback, interpolation
  scope, cell statistics, and query-label independence.

## Intentionally excluded

- audio datasets and derived embedding caches;
- encoder checkpoints and third-party model weights;
- full `analysis_output/paper_experiments/` (approximately 1 GB);
- machine-specific paths embedded in the local audit archive;
- failed exploratory methods that are not part of final ProtoFill.

## Before publishing

```bash
python -m unittest discover -s tests -v
python -m protofill --help
python -m experiments.select_loco --help
python -m experiments.evaluate --help
git diff --check
git status --short
```

Inspect the staged file list before pushing:

```bash
git diff --cached --stat
git diff --cached
```

The repository currently has no explicit open-source license. Select a license
with all authors before making the repository public. Never place access
tokens in a Git remote URL; use a credential helper or SSH authentication.
