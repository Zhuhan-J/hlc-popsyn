"""Command-line runner for end-to-end hierarchical mixture synthesis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io import load_csv, prepare_from_merged_table
from .pipeline import fit_and_generate


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit EM and generate synthetic population from merged demographics CSV"
    )
    p.add_argument("--input", required=True, help="Path to merged household-individual CSV")
    p.add_argument("--output", required=True, help="Output .npz file path")
    p.add_argument("--G", type=int, required=True, help="Number of household latent classes")
    p.add_argument("--M", type=int, required=True, help="Number of individual latent classes")
    p.add_argument(
        "--n-households",
        type=int,
        required=True,
        help="Number of synthetic households to generate",
    )
    p.add_argument("--n-restarts", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--concentration", type=float, default=1.0)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--no-rejection", action="store_true", help="Skip two-person rejection correction"
    )
    p.add_argument(
        "--initial-oversample",
        type=float,
        default=1.5,
        help="Multiplier for initial draw size when rejection is enabled (>=1.0)",
    )
    p.add_argument(
        "--topup-factor",
        type=float,
        default=1.7,
        help="Multiplier for top-up batch size when rejection leaves a shortfall",
    )
    p.add_argument(
        "--max-topup-iters",
        type=int,
        default=10,
        help="Maximum number of top-up batches before raising on shortfall",
    )
    return p


# paper_ref: Section 4.2-4.4 execution chain (fit, draw, reject)
# matlab_ref: run_multilevel_1.m + run_create_2.m + run_two_within_the_same_4.m
# intent: user-facing one-command reproduction workflow.
def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    raw = load_csv(input_path, skip_header=True)
    prepared = prepare_from_merged_table(raw)

    result = fit_and_generate(
        grpdata=prepared.grpdata,
        indgid=prepared.indgid,
        inddata=prepared.inddata,
        G=args.G,
        M=args.M,
        n_households=args.n_households,
        n_restarts=args.n_restarts,
        max_iter=args.max_iter,
        concentration=args.concentration,
        tol=args.tol,
        seed=args.seed,
        apply_rejection=not args.no_rejection,
        initial_oversample=args.initial_oversample,
        topup_factor=args.topup_factor,
        max_topup_iters=args.max_topup_iters,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pi_g=result.em_result.pi_g,
        pi_m=result.em_result.pi_m,
        log_phi_g=np.array(result.em_result.log_phi_g, dtype=object),
        log_phi_m=np.array(result.em_result.log_phi_m, dtype=object),
        household_data=result.synthetic_final.household_data,
        individual_data=result.synthetic_final.individual_data,
        individual_group_id=result.synthetic_final.individual_group_id,
        household_class=result.synthetic_final.household_class,
        individual_class=result.synthetic_final.individual_class,
        likelihood=result.em_result.likelihood,
        converged=np.array([result.em_result.converged], dtype=bool),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
