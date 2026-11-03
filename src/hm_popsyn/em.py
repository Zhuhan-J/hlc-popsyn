"""EM and model-selection utilities for hierarchical mixture replication.

This module implements the universal individual-class EM algorithm used in
EM_multilevel_across.m and utilities for BIC-based model selection.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .kernels import logsumexp, sum_eik


@dataclass(slots=True)
class ModelSelectionResult:
    """Store model selection metadata for one (G, M) configuration."""

    G: int
    M: int
    best_log_likelihood: float
    bic: float


@dataclass(slots=True)
class EMResult:
    """Fitted EM parameters and diagnostics.

    Attributes
    ----------
    pi_g:
        Household class weights, shape (G,).
    pi_m:
        Conditional individual class weights, shape (G, M).
    log_phi_g:
        Household attribute logits by class, each entry shape (d_k, G).
    log_phi_m:
        Individual attribute logits by universal class, each entry shape (d_k, M).
    wj:
        Household responsibilities, shape (N_household, G).
    x_jik:
        Individual responsibilities per household class and individual class,
        shape (N_individual, G, M).
    likelihood:
        Log-likelihood trace.
    converged:
        Whether relative likelihood improvement reached tolerance.
    """

    pi_g: np.ndarray
    pi_m: np.ndarray
    log_phi_g: list[np.ndarray]
    log_phi_m: list[np.ndarray]
    wj: np.ndarray
    x_jik: np.ndarray
    likelihood: np.ndarray
    converged: bool


# paper_ref: Section 4.2, Eq. (14)
# matlab_ref: run_multilevel_1.m:88
# intent: compute Bayesian Information Criterion used for G/M selection.
def bic_score(best_log_likelihood: float, n_params: int, n_individuals: int) -> float:
    """Compute BIC = -2 log L + D_f log(n_individuals)."""
    if n_individuals <= 0:
        raise ValueError("n_individuals must be positive")
    return float(-2.0 * best_log_likelihood + n_params * np.log(n_individuals))


# paper_ref: Section 3.4 parameter counting discussion + Section 4.2 Eq. (14)
# matlab_ref: run_multilevel_1.m:87-88
# intent: count free parameters for universal individual classes.
def count_parameters_universal(G: int, M: int, grpcate: np.ndarray, indcate: np.ndarray) -> int:
    """Count free parameters for universal-class hierarchical mixture model."""
    grpc = np.asarray(grpcate, dtype=int)
    indc = np.asarray(indcate, dtype=int)
    return int((G - 1) + G * (M - 1) + np.sum(grpc - 1) * G + np.sum(indc - 1) * M)


def _normalize_rows(x: np.ndarray, eps: float) -> np.ndarray:
    row_sum = np.sum(x, axis=1, keepdims=True)
    return x / np.maximum(row_sum, eps)


def _as_category_matrix(name: str, arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=int)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if np.any(out < 1):
        raise ValueError(f"{name} must be 1-based categorical values")
    return out


def _prepare_group_index(indgid: np.ndarray, ngrp: int) -> np.ndarray:
    gid = np.asarray(indgid, dtype=int).ravel()
    if gid.ndim != 1:
        raise ValueError("indgid must be 1D")
    if np.any(gid < 1) or np.any(gid > ngrp):
        raise ValueError("indgid values must be in 1..number_of_households")
    return gid - 1


def _init_log_phi(
    n_categories: np.ndarray, n_classes: int, concentration: float, rng: np.random.Generator
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for n_cat in n_categories:
        temp = rng.gamma(shape=1.0, scale=concentration, size=(int(n_cat), n_classes))
        temp = temp / np.maximum(temp.sum(axis=0, keepdims=True), 1e-15)
        out.append(np.log(np.maximum(temp, 1e-300)))
    return out


# paper_ref: Section 3.3 Eq. (8)-(13), Section 3.4 Eq. (13)
# matlab_ref: EM_multilevel_across.m:86-135
# intent: full universal-class EM implementation used as the main training routine.
def em_multilevel_across_universal(
    # grpdata: household-level categorical attributes, shape (n_households, n_household_attrs).
    grpdata: np.ndarray,
    # indgid: household index of each individual, shape (n_individuals,); 1-based household
    # ids corresponding to rows in grpdata.
    indgid: np.ndarray,
    # inddata: individual-level categorical attributes, shape (n_individuals, n_individual_attrs).
    inddata: np.ndarray,
    G: int,
    M: int,
    max_iter: int = 4000,
    concentration: float = 1.0,
    tol: float = 1e-10,
    rng: np.random.Generator | None = None,
    init: dict | None = None,
    eps: float = 1e-12,
) -> EMResult:
    """Fit universal-class hierarchical mixture model using EM.

    Parameters follow MATLAB EM_multilevel_across.m semantics except tensor
    initialization path, which is intentionally omitted in this first complete
    Python port to keep dependencies minimal and deterministic.
    """
    if G <= 0 or M <= 0:
        raise ValueError("G and M must be positive")
    if max_iter <= 1:
        raise ValueError("max_iter must be greater than 1")

    rng = rng or np.random.default_rng()

    grpdata_c = _as_category_matrix("grpdata", grpdata)
    inddata_c = _as_category_matrix("inddata", inddata)

    ngrp = grpdata_c.shape[0]
    nind = inddata_c.shape[0]
    if nind == 0 or ngrp == 0:
        raise ValueError("grpdata and inddata must be non-empty")

    gid0 = _prepare_group_index(indgid, ngrp)

    grpcate = np.max(grpdata_c, axis=0)
    indcate = np.max(inddata_c, axis=0)

    numgat = grpdata_c.shape[1]
    numatt = inddata_c.shape[1]

    log_phi_m = _init_log_phi(indcate, M, concentration, rng)
    log_phi_g = _init_log_phi(grpcate, G, concentration, rng)

    pi_g = rng.gamma(shape=1.0, scale=concentration, size=G)
    pi_g = pi_g / np.maximum(np.sum(pi_g), eps)

    pi_m = rng.gamma(shape=1.0, scale=concentration, size=(G, M))
    pi_m = _normalize_rows(pi_m, eps)

    if init is not None:
        if "pi_g" in init:
            pi_g = np.asarray(init["pi_g"], dtype=float).ravel()
            pi_g = pi_g / np.maximum(np.sum(pi_g), eps)
        if "pi_m" in init:
            pi_m = np.asarray(init["pi_m"], dtype=float)
            if pi_m.shape != (G, M):
                raise ValueError("init['pi_m'] must have shape (G, M)")
            pi_m = _normalize_rows(pi_m, eps)
        if "log_phi_g" in init:
            log_phi_g = [np.asarray(x, dtype=float) for x in init["log_phi_g"]]
        if "log_phi_m" in init:
            log_phi_m = [np.asarray(x, dtype=float) for x in init["log_phi_m"]]

    lik = np.zeros(max_iter, dtype=float)
    rel = np.inf

    wj = np.zeros((ngrp, G), dtype=float)
    x_jik = np.zeros((nind, G, M), dtype=float)

    converged = False
    n_iter_done = 0

    for it in range(max_iter):
        # E-step: build log probability of individual attributes per universal class.
        f_ind = np.zeros((nind, M), dtype=float)
        for att in range(numatt):
            f_ind += log_phi_m[att][inddata_c[:, att] - 1, :]

        h = np.exp(f_ind)[:, None, :] * pi_m[None, :, :]
        denom_h = np.maximum(np.sum(h, axis=2, keepdims=True), eps)
        x_jik = h / denom_h
        log_g_ji = np.log(np.maximum(np.sum(h, axis=2), eps))

        tt0 = np.zeros((ngrp, G), dtype=float)
        for g in range(G):
            np.add.at(tt0[:, g], gid0, log_g_ji[:, g])

        temp_g = np.zeros((ngrp, G), dtype=float)
        for att in range(numgat):
            temp_g += log_phi_g[att][grpdata_c[:, att] - 1, :]

        temp = temp_g + tt0 + np.log(np.maximum(pi_g[None, :], eps))
        temp_max = np.max(temp, axis=1, keepdims=True)
        wj0 = np.exp(temp - temp_max)
        wj = wj0 / np.maximum(np.sum(wj0, axis=1, keepdims=True), eps)

        # M-step: update pi_m using weighted responsibilities.
        second = x_jik * wj[gid0, :, None]
        pi_m = np.sum(second, axis=0) + eps
        pi_m = _normalize_rows(pi_m, eps)

        # M-step: update pi_g.
        pi_g = np.sum(wj, axis=0) + eps
        pi_g = pi_g / np.maximum(np.sum(pi_g), eps)

        # M-step: update household attribute distributions.
        for att in range(numgat):
            tt = sum_eik(wj, grpdata_c[:, att], int(grpcate[att])) + eps
            log_phi_g[att] = np.log(tt / np.maximum(np.sum(tt, axis=0, keepdims=True), eps))

        # M-step: update universal individual attribute distributions.
        second2 = np.sum(second, axis=1)
        for att in range(numatt):
            tt = sum_eik(second2, inddata_c[:, att], int(indcate[att])) + eps
            log_phi_m[att] = np.log(tt / np.maximum(np.sum(tt, axis=0, keepdims=True), eps))

        # Log-likelihood (mirrors MATLAB structure; recomputed from temp and current pi_g).
        temp_lik = temp_g + tt0 + np.log(np.maximum(pi_g[None, :], eps))
        lik[it] = float(np.sum(logsumexp(temp_lik, axis=1)))
        n_iter_done = it + 1

        if it >= 1:
            rel = abs((lik[it] - lik[it - 1]) / max(abs(lik[it - 1]), eps))
            if rel <= tol:
                converged = True
                break

    likelihood = lik[:n_iter_done]
    return EMResult(
        pi_g=pi_g,
        pi_m=pi_m,
        log_phi_g=log_phi_g,
        log_phi_m=log_phi_m,
        wj=wj,
        x_jik=x_jik,
        likelihood=likelihood,
        converged=converged,
    )


# paper_ref: Section 4.2, Eq. (14)
# matlab_ref: run_multilevel_1.m:85-90
# intent: run multi-restart EM for one (G, M) and return best model by BIC.
def fit_em_with_restarts(
    grpdata: np.ndarray,
    indgid: np.ndarray,
    inddata: np.ndarray,
    G: int,
    M: int,
    n_restarts: int = 20,
    max_iter: int = 4000,
    concentration: float = 1.0,
    tol: float = 1e-10,
    seed: int | None = None,
) -> tuple[EMResult, ModelSelectionResult]:
    """Run multiple EM restarts and select model by best BIC."""
    if n_restarts <= 0:
        raise ValueError("n_restarts must be positive")

    base_rng = np.random.default_rng(seed)

    grpdata_c = _as_category_matrix("grpdata", grpdata)
    inddata_c = _as_category_matrix("inddata", inddata)
    grpcate = np.max(grpdata_c, axis=0)
    indcate = np.max(inddata_c, axis=0)

    n_params = count_parameters_universal(G, M, grpcate, indcate)
    n_ind = inddata_c.shape[0]

    best_res: EMResult | None = None
    best_bic = np.inf
    best_ll = -np.inf

    for _ in range(n_restarts):
        child_rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
        res = em_multilevel_across_universal(
            grpdata=grpdata_c,
            indgid=indgid,
            inddata=inddata_c,
            G=G,
            M=M,
            max_iter=max_iter,
            concentration=concentration,
            tol=tol,
            rng=child_rng,
        )
        ll = float(np.max(res.likelihood))
        bic = bic_score(ll, n_params, n_ind)
        if bic < best_bic:
            best_bic = bic
            best_ll = ll
            best_res = res

    assert best_res is not None
    return best_res, ModelSelectionResult(
        G=G, M=M, best_log_likelihood=best_ll, bic=float(best_bic)
    )


# paper_ref: Section 4.2, Eq. (14)
# matlab_ref: run_multilevel_1.m:61-67
# intent: grid-search over G/M choices and pick global best by BIC.
def select_model_grid(
    grpdata: np.ndarray,
    indgid: np.ndarray,
    inddata: np.ndarray,
    g_values: Sequence[int],
    m_values: Sequence[int],
    n_restarts: int = 5,
    max_iter: int = 1000,
    concentration: float = 1.0,
    tol: float = 1e-8,
    seed: int | None = None,
) -> tuple[EMResult, ModelSelectionResult, list[ModelSelectionResult]]:
    """Run BIC grid search across candidate (G, M) pairs."""
    results: list[ModelSelectionResult] = []
    best_model: EMResult | None = None
    best_summary: ModelSelectionResult | None = None

    base_rng = np.random.default_rng(seed)
    for G in g_values:
        for M in m_values:
            child_seed = int(base_rng.integers(0, 2**32 - 1))
            em_res, summary = fit_em_with_restarts(
                grpdata=grpdata,
                indgid=indgid,
                inddata=inddata,
                G=int(G),
                M=int(M),
                n_restarts=n_restarts,
                max_iter=max_iter,
                concentration=concentration,
                tol=tol,
                seed=child_seed,
            )
            results.append(summary)
            if best_summary is None or summary.bic < best_summary.bic:
                best_summary = summary
                best_model = em_res

    assert best_model is not None and best_summary is not None
    return best_model, best_summary, results
