"""Public, encoder-agnostic implementation of ProtoFill."""

from .core import (
    DEFAULT_RIDGE_LAMBDA,
    DEFAULT_STRENGTH,
    CellTable,
    ProtoFillFit,
    build_cell_table,
    build_domain_prototypes,
    fit_protofill,
    fit_two_way_ridge,
    predict,
    support_connected,
)

__all__ = [
    "DEFAULT_RIDGE_LAMBDA",
    "DEFAULT_STRENGTH",
    "CellTable",
    "ProtoFillFit",
    "build_cell_table",
    "build_domain_prototypes",
    "fit_protofill",
    "fit_two_way_ridge",
    "predict",
    "support_connected",
]
