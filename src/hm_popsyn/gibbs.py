"""Gibbs sampler for universal-class hierarchical mixture model.

This implementation follows the same model structure as Gibbs_universal.m:
- household class assignment per household
- universal individual class assignment per individual
- Dirichlet-Multinomial conjugate updates
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GibbsResult:
    """Posterior summaries and traces from Gibbs sampling."""

    pi_g_mean: np.ndarray
    pi_m_mean: np.ndarray
    phi_g_mean: list[np.ndarray]
    phi_m_mean: list[np.ndarray]
    z_trace: np.ndarray
    c_trace: np.ndarray


def _as_category_matrix(name: str, arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=int)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if np.any(out < 1):
        raise ValueError(f"{name} must be 1-based categorical values")
    return out


def _prepare_group_index(indgid: np.ndarray, ngrp: int) -> np.ndarray:
    gid = np.asarray(indgid, dtype=int).ravel()
    if np.any(gid < 1) or np.any(gid > ngrp):
        raise ValueError("indgid values must be in 1..number_of_households")
    return gid - 1


def _sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> int:
    p = np.asarray(probs, dtype=float)
    p = np.maximum(p, 1e-300)
    p = p / np.sum(p)
    return int(rng.choice(np.arange(p.size), p=p))


def _logsumexp_1d(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def _sample_parameters(
    z: np.ndarray,
    c: np.ndarray,
    grpdata: np.ndarray,
    inddata: np.ndarray,
    indgid0: np.ndarray,
    G: int,
    M: int,
    concentration: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    numgat = grpdata.shape[1]
    numatt = inddata.shape[1]

    grpcate = np.max(grpdata, axis=0)
    indcate = np.max(inddata, axis=0)

    z_count = np.bincount(z, minlength=G)
    pi_g = rng.dirichlet(np.full(G, concentration) + z_count)

    pi_m = np.zeros((G, M), dtype=float)
    for g in range(G):
        idx = np.where(z[indgid0] == g)[0]
        if idx.size == 0:
            pi_m[g, :] = rng.dirichlet(np.full(M, concentration))
        else:
            c_count = np.bincount(c[idx], minlength=M)
            pi_m[g, :] = rng.dirichlet(np.full(M, concentration) + c_count)

    phi_g: list[np.ndarray] = []
    for att in range(numgat):
        d = int(grpcate[att])
        mat = np.zeros((d, G), dtype=float)
        col = grpdata[:, att] - 1
        for g in range(G):
            idx = np.where(z == g)[0]
            count = np.zeros(d, dtype=float)
            if idx.size > 0:
                np.add.at(count, col[idx], 1.0)
            mat[:, g] = rng.dirichlet(np.full(d, concentration) + count)
        phi_g.append(mat)

    phi_m: list[np.ndarray] = []
    for att in range(numatt):
        d = int(indcate[att])
        mat = np.zeros((d, M), dtype=float)
        col = inddata[:, att] - 1
        for m in range(M):
            idx = np.where(c == m)[0]
            count = np.zeros(d, dtype=float)
            if idx.size > 0:
                np.add.at(count, col[idx], 1.0)
            mat[:, m] = rng.dirichlet(np.full(d, concentration) + count)
        phi_m.append(mat)

    return pi_g, pi_m, phi_g, phi_m


# paper_ref: Section 3.3 and Section 4.2 (Bayesian estimation variant)
# matlab_ref: Gibbs_universal.m:58-142
# intent: full Gibbs sampler for universal individual classes.
def gibbs_universal(
    grpdata: np.ndarray,
    indgid: np.ndarray,
    inddata: np.ndarray,
    G: int,
    M: int,
    n_iter: int = 800,
    burn_in: int = 300,
    thin: int = 5,
    concentration: float = 1.0,
    seed: int | None = None,
) -> GibbsResult:
    """Run Gibbs sampling and return posterior mean parameters.

    Parameters use 1-based categorical inputs consistent with MATLAB code.
    """
    if G <= 0 or M <= 0:
        raise ValueError("G and M must be positive")
    if n_iter <= 0 or burn_in < 0 or thin <= 0:
        raise ValueError("invalid n_iter/burn_in/thin")

    rng = np.random.default_rng(seed)

    grp = _as_category_matrix("grpdata", grpdata)
    ind = _as_category_matrix("inddata", inddata)

    ngrp = grp.shape[0]
    nind = ind.shape[0]
    gid0 = _prepare_group_index(indgid, ngrp)

    z = rng.integers(0, G, size=ngrp)
    c = rng.integers(0, M, size=nind)

    pi_g, pi_m, phi_g, phi_m = _sample_parameters(z, c, grp, ind, gid0, G, M, concentration, rng)

    z_samples: list[np.ndarray] = []
    c_samples: list[np.ndarray] = []
    pi_g_samples: list[np.ndarray] = []
    pi_m_samples: list[np.ndarray] = []
    phi_g_samples: list[list[np.ndarray]] = []
    phi_m_samples: list[list[np.ndarray]] = []

    numgat = grp.shape[1]
    numatt = ind.shape[1]

    for it in range(n_iter):
        # Sample household class labels z_j.
        for j in range(ngrp):
            idx = np.where(gid0 == j)[0]
            logp = np.log(np.maximum(pi_g, 1e-300))
            for g in range(G):
                for att in range(numgat):
                    x = grp[j, att] - 1
                    logp[g] += np.log(np.maximum(phi_g[att][x, g], 1e-300))
                for i in idx:
                    log_mix = np.zeros(M, dtype=float)
                    for m in range(M):
                        val = np.log(np.maximum(pi_m[g, m], 1e-300))
                        for att in range(numatt):
                            x = ind[i, att] - 1
                            val += np.log(np.maximum(phi_m[att][x, m], 1e-300))
                        log_mix[m] = val
                    logp[g] += _logsumexp_1d(log_mix)
            z[j] = _sample_categorical(np.exp(logp - np.max(logp)), rng)

        # Sample individual class labels c_ji.
        for i in range(nind):
            g = z[gid0[i]]
            logp = np.log(np.maximum(pi_m[g, :], 1e-300))
            for m in range(M):
                for att in range(numatt):
                    x = ind[i, att] - 1
                    logp[m] += np.log(np.maximum(phi_m[att][x, m], 1e-300))
            c[i] = _sample_categorical(np.exp(logp - np.max(logp)), rng)

        # Sample parameters conditional on assignments.
        pi_g, pi_m, phi_g, phi_m = _sample_parameters(
            z, c, grp, ind, gid0, G, M, concentration, rng
        )

        # Store posterior samples after burn-in and thinning.
        if it >= burn_in and ((it - burn_in) % thin == 0):
            z_samples.append(z.copy())
            c_samples.append(c.copy())
            pi_g_samples.append(pi_g.copy())
            pi_m_samples.append(pi_m.copy())
            phi_g_samples.append([x.copy() for x in phi_g])
            phi_m_samples.append([x.copy() for x in phi_m])

    if not pi_g_samples:
        raise ValueError("No posterior samples collected; adjust burn_in/thin/n_iter")

    pi_g_mean = np.mean(np.stack(pi_g_samples, axis=0), axis=0)
    pi_m_mean = np.mean(np.stack(pi_m_samples, axis=0), axis=0)

    phi_g_mean: list[np.ndarray] = []
    for att in range(numgat):
        stack = np.stack([s[att] for s in phi_g_samples], axis=0)
        phi_g_mean.append(np.mean(stack, axis=0))

    phi_m_mean: list[np.ndarray] = []
    for att in range(numatt):
        stack = np.stack([s[att] for s in phi_m_samples], axis=0)
        phi_m_mean.append(np.mean(stack, axis=0))

    return GibbsResult(
        pi_g_mean=pi_g_mean,
        pi_m_mean=pi_m_mean,
        phi_g_mean=phi_g_mean,
        phi_m_mean=phi_m_mean,
        z_trace=np.stack(z_samples, axis=0),
        c_trace=np.stack(c_samples, axis=0),
    )
