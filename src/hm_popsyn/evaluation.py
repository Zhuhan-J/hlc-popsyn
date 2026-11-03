"""Evaluation utilities for synthetic population quality checks."""

from __future__ import annotations

import numpy as np


# paper_ref: Section 4.3/4.4, Eq. (15)
# matlab_ref: run_create_2.m:93-143 and run_two_within_the_same_4.m:15-23
# intent: compute Cramer's V from a 2D contingency table.
def cramer_v_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Cramer's V for a contingency table."""
    table = np.asarray(counts, dtype=float)
    if table.ndim != 2:
        raise ValueError("counts must be 2D")

    total = table.sum()
    if total <= 0:
        return 0.0

    p = table / total
    p = (p + eps) / np.sum(p + eps)
    row = p.sum(axis=1, keepdims=True)
    col = p.sum(axis=0, keepdims=True)
    prod = row @ col

    chi_part = np.sum(((p - prod) ** 2) / np.maximum(prod, eps))
    denom = min(table.shape[0] - 1, table.shape[1] - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi_part / denom))


# matlab_ref: run_draw_figures_3.m:43-67, 108-291
# notes: returns normalized marginal frequency for a 1-based categorical column.
def marginal_distribution(col: np.ndarray, n_categories: int | None = None) -> np.ndarray:
    """Compute normalized marginal distribution for 1-based coded categories."""
    values = np.asarray(col, dtype=int).ravel()
    if values.size == 0:
        return np.zeros(0, dtype=float)
    if n_categories is None:
        n_categories = int(np.max(values))
    out = np.zeros(int(n_categories), dtype=float)
    for v in values:
        if v < 1 or v > n_categories:
            raise ValueError(f"category out of range: {v}")
        out[v - 1] += 1.0
    return out / out.sum()
