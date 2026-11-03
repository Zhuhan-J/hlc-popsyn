"""End-to-end pipeline: fit EM, synthesize population, and apply rejection stage."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .em import EMResult, fit_em_with_restarts
from .synthesis_eval import (
    SynthesisResult,
    generate_synthetic_population,
    generate_with_multi_size_rejection,
)


@dataclass(slots=True)
class PipelineResult:
    """Main end-to-end output container."""

    em_result: EMResult
    synthetic_raw: SynthesisResult
    synthetic_final: SynthesisResult


# paper_ref: Section 4.2 -> 4.4, Eq. (14), Eq. (3), Eq. (16)
# matlab_ref: run_multilevel_1.m, run_create_2.m, run_two_within_the_same_4.m
# intent: complete runnable workflow from observed data to final synthetic population.
def fit_and_generate(
    grpdata: np.ndarray,
    indgid: np.ndarray,
    inddata: np.ndarray,
    G: int,
    M: int,
    n_households: int,
    n_restarts: int = 5,
    max_iter: int = 1000,
    concentration: float = 1.0,
    tol: float = 1e-8,
    seed: int | None = None,
    apply_rejection: bool = True,
    age_col: int = 0,
    employ_col: int = 1,
    sex_col: int | None = None,
    household_size_col: int = 3,
    initial_oversample: float = 1.5,
    topup_factor: float = 1.7,
    max_topup_iters: int = 10,
    age_code_to_order: np.ndarray | None = None,
    employ_code_to_group: np.ndarray | None = None,
    min_target_samples: int = 30,
) -> PipelineResult:
    """Run fit -> synthetic generation -> optional rejection correction.

    When ``apply_rejection=True`` the rejection step uses the oversample-then-redraw
    strategy in ``generate_with_multi_size_rejection`` so the final population contains
    exactly ``n_households`` rows even when rejection drops a large fraction of
    any household size. The ``initial_oversample`` / ``topup_factor`` /
    ``max_topup_iters`` knobs control that behaviour.
    """
    if sex_col is not None:
        warnings.warn(
            "sex_col is deprecated; use employ_col for age x employ rejection.",
            DeprecationWarning,
            stacklevel=2,
        )
        employ_col = sex_col
    em_result, _ = fit_em_with_restarts(
        grpdata=grpdata,
        indgid=indgid,
        inddata=inddata,
        G=G,
        M=M,
        n_restarts=n_restarts,
        max_iter=max_iter,
        concentration=concentration,
        tol=tol,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    phi_g_lin = [np.exp(x) for x in em_result.log_phi_g]
    phi_m_lin = [np.exp(x) for x in em_result.log_phi_m]

    raw = generate_synthetic_population(
        n_households=n_households,
        pi_g=em_result.pi_g,
        pi_m=em_result.pi_m,
        phi_g=phi_g_lin,
        phi_m=phi_m_lin,
        household_size_col=household_size_col,
        rng=rng,
    )

    if apply_rejection:
        # Use a separate Generator so the raw draw above stays reproducible per seed.
        rej_rng = np.random.default_rng(seed + 1 if seed is not None else None)
        final = generate_with_multi_size_rejection(
            n_households=n_households,
            pi_g=em_result.pi_g,
            pi_m=em_result.pi_m,
            phi_g=phi_g_lin,
            phi_m=phi_m_lin,
            target_individual_data=inddata,
            target_individual_group_id=indgid,
            household_size_col=household_size_col,
            age_col=age_col,
            employ_col=employ_col,
            initial_oversample=initial_oversample,
            topup_factor=topup_factor,
            max_topup_iters=max_topup_iters,
            rng=rej_rng,
            age_code_to_order=age_code_to_order,
            employ_code_to_group=employ_code_to_group,
            min_target_samples=min_target_samples,
        )
    else:
        final = raw

    return PipelineResult(em_result=em_result, synthetic_raw=raw, synthetic_final=final)
