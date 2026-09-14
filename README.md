# Regularized Additive Prototype Completion (RAPC) 

RAPC is a closed-form inference module for completing a class prototype
that is missing from a **known** domain. It is encoder-agnostic: any acoustic
encoder can be used as long as it exports embeddings with class and domain
labels for the training split.

This repository intentionally contains only the final RAPC method and the
experiments needed to select and evaluate it. It does not contain the original
competition training system, failed exploratory methods, audio datasets,
checkpoints, embedding caches, or paper source.

## Final configuration

```text
ridge lambda       = 1
interpolation g    = 0.75
geometry           = normalized direction interpolation
cell weighting     = equal
```

Parameters are selected using training/validation embeddings only. Evaluation
labels are read only after prediction for metric computation; they never enter
prototype construction.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Minimal API

```python
from rapc import fit_rapc, predict

fitted = fit_rapc(train_embeddings, train_labels, train_domains)
predictions = predict(fitted, query_embeddings, query_domains)
```

See [rapc/README.md](rapc/README.md) for the equations and API, and
[experiments/README.md](experiments/README.md) for cache schemas and commands.

## Repository layout

```text
rapc/       final method and embedding-cache CLI
experiments/     validation-only LOCO selection and held-cell evaluation
tests/           synthetic protocol and numerical tests
```

## Verification

```bash
python -m unittest discover -s tests -v
python -m rapc --help
python -m experiments.select_loco --help
python -m experiments.select_loco_multiseed --help
python -m experiments.evaluate --help
```

## License

This project is released under the [MIT License](LICENSE).
