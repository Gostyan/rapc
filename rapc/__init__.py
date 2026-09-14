"""Public, encoder-agnostic implementation of RAPC."""

from .core import (
    DEFAULT_RIDGE_LAMBDA,
    DEFAULT_STRENGTH,
    CellTable,
    RAPCFit,
    build_cell_table,
    build_domain_prototypes,
    fit_rapc,
    fit_two_way_ridge,
    predict,
    support_connected,
)

__all__ = [
    "DEFAULT_RIDGE_LAMBDA",
    "DEFAULT_STRENGTH",
    "CellTable",
    "RAPCFit",
    "build_cell_table",
    "build_domain_prototypes",
    "fit_rapc",
    "fit_two_way_ridge",
    "predict",
    "support_connected",
]
