"""Low-level kernels translated from MATLAB/MEX helpers.

Annotation policy:
- matlab_ref: original MATLAB/C source and purpose.
- notes: translation caveats and indexing behavior.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


# matlab_ref: int_hist.c:1-47
# notes: MATLAB inputs are 1-based category indices; output is counts for 1..n.
def int_hist(x: np.ndarray, n_bins: int | None = None) -> np.ndarray:
    """Histogram integer-coded categories in 1..n_bins.

    Behavior follows int_hist.c:
    - accepts values expected to be integer-like
    - uses 1-based indexing semantics
    - raises ValueError on out-of-bounds values
    """
    values = np.asarray(x, dtype=float).ravel()
    if values.size == 0:
        if n_bins is None:
            return np.zeros(0, dtype=float)
        return np.zeros(int(n_bins), dtype=float)

    if n_bins is None:
        n_bins = int(np.max(values))
    n_bins = int(n_bins)

    out = np.zeros(n_bins, dtype=float)
    for val in values:
        v = int(val) - 1
        if v < 0 or v >= n_bins:
            raise ValueError(f"value out of bounds: {val}")
        out[v] += 1.0
    return out


# matlab_ref: sum_eik.c:1-25
# notes: mask is interpreted as 1-based bin index; output shape is (bins, nfac).
def sum_eik(eik: np.ndarray, mask: np.ndarray, bins: int) -> np.ndarray:
    """
    eik = element-wise probabilities for each individual and category;
    shape (n_individuals, n_factors).
    mask = category index for each individual; shape (n_individuals,).
    bins = total number of categories (bins) to aggregate into.

    The mask values are expected to be 1-based indices that indicate which bin each
    row of eik belongs to. The output is a matrix where each row corresponds to a bin
    and contains the sum of the rows of eik that belong to that bin.
    Aggregate rows of eik by 1-based mask categories.

    Equivalent to MATLAB MEX function sum_eik used in EM updates.
    """
    eik_arr = np.asarray(eik, dtype=float)
    if eik_arr.ndim != 2:
        raise ValueError("eik must be a 2D array")
    mask_arr = np.asarray(mask, dtype=int).ravel()
    if eik_arr.shape[0] != mask_arr.size:
        raise ValueError("mask length must match eik rows")

    bins = int(bins)
    out = np.zeros((bins, eik_arr.shape[1]), dtype=float)
    for i, w1 in enumerate(mask_arr):
        w = int(w1) - 1
        if w < 0 or w >= bins:
            raise ValueError(f"mask value out of bounds: {w1}")
        out[w, :] += eik_arr[i, :]
    return out


# matlab_ref: mnrnd_new.m:1-10
# notes: returns one-hot rows sampled from row-wise categorical probabilities.
def mnrnd_new(prob_mat: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample one-hot categorical outcomes row-wise.
    mnrnd = multinomial random sampling; here we sample one category per row based on
    the provided probabilities.
    Parameters
    ----------
    prob_mat:
        (n, d) probability matrix; rows may be unnormalized positive values.
    rng:
        Optional numpy Generator for deterministic sampling.
    """
    probs = np.asarray(prob_mat, dtype=float)
    if probs.ndim != 2:
        raise ValueError("prob_mat must be 2D")
    if np.any(probs < 0):
        raise ValueError("prob_mat cannot contain negative values")

    rng = rng or np.random.default_rng()
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("each row of prob_mat must have positive sum")
    p = probs / row_sums

    cum = np.cumsum(p, axis=1)
    u = rng.random((p.shape[0], 1))
    idx = (u > cum).sum(axis=1)

    out = np.zeros_like(p, dtype=int)
    out[np.arange(p.shape[0]), idx] = 1
    return out


# matlab_ref: logsumexp.m:1-23
# notes: default axis in MATLAB file is dim=1 (column-wise); here axis defaults to 0.
def logsumexp(a: np.ndarray, axis: int = 0, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log(sum(exp(a), axis))."""
    arr = np.asarray(a, dtype=float)
    a_max = np.max(arr, axis=axis, keepdims=True)
    shifted = arr - a_max
    s = a_max + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))

    if not keepdims:
        s = np.squeeze(s, axis=axis)
    return s


# matlab_ref: sumdims.m:1-8
# notes: dims are expected in MATLAB 1-based indexing.
def sumdims(r: np.ndarray, dims: Iterable[int]) -> np.ndarray:
    """Sequentially sum across MATLAB-style dimensions and squeeze result."""
    out = np.asarray(r, dtype=float)
    for dim in dims:
        axis = int(dim) - 1
        out = np.sum(out, axis=axis, keepdims=True)
    return np.squeeze(out)
