"""Synthetic population generation and postprocessing.

Given model parameters from EM, generate synthetic households and individuals,
then apply the acceptance-rejection correction by household size.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _build_code_to_index_map(ordered_codes: Sequence[int], max_code: int) -> np.ndarray:
    mapping = np.full(max_code + 1, -1, dtype=int)
    for idx, code in enumerate(ordered_codes):
        mapping[int(code)] = int(idx)
    return mapping


# ---------------------------------------------------------------------------
# Default lookup tables for the rejection features.
# They assume one specific integer coding of the age and employment columns.
# For any other coding, pass ``age_code_to_order`` and ``employ_code_to_group``.
# ---------------------------------------------------------------------------
# Age column: codes 1..18 are already in ordinal order (code k -> position k - 1).
# AGE_DIFF_MAX is the maximum possible ordinal difference between two age codes.
AGE_CODE_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
AGE_DIFF_MAX = len(AGE_CODE_ORDER) - 1  # = 17
AGE_CODE_TO_ORDER = _build_code_to_index_map(AGE_CODE_ORDER, max_code=18)

# Employment column: code -> coarse group index (0..5); unlisted codes map to -1 (invalid).
EMPLOY_CODE_TO_GROUP = np.full(12, -1, dtype=int)
EMPLOY_CODE_TO_GROUP[1] = 0
EMPLOY_CODE_TO_GROUP[2] = 0
EMPLOY_CODE_TO_GROUP[9] = 0
EMPLOY_CODE_TO_GROUP[6] = 0
EMPLOY_CODE_TO_GROUP[3] = 1
EMPLOY_CODE_TO_GROUP[4] = 2
EMPLOY_CODE_TO_GROUP[8] = 3
EMPLOY_CODE_TO_GROUP[7] = 4
EMPLOY_CODE_TO_GROUP[10] = 5
EMPLOY_CODE_TO_GROUP[5] = 5
EMPLOY_CODE_TO_GROUP[11] = 5


@dataclass(slots=True)
class SynthesisResult:
    """Container for generated synthetic population.

    household_data:
        Shape (N_household, H), categories coded as 1..d_k.
    individual_data:
        Shape (N_individual, I), categories coded as 1..d_k.
    individual_group_id:
        Shape (N_individual,), household id in 1-based indexing.
    household_class:
        Shape (N_household,), sampled household latent classes in 1..G.
    individual_class:
        Shape (N_individual,), sampled individual latent classes in 1..M.
    """

    household_data: np.ndarray
    individual_data: np.ndarray
    individual_group_id: np.ndarray
    household_class: np.ndarray
    individual_class: np.ndarray


def _group_members(individual_group_id: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return sorted unique group ids and index lists for each group."""
    gid = np.asarray(individual_group_id, dtype=int).ravel()
    groups = np.unique(gid)
    members: list[np.ndarray] = []
    for g in groups:
        members.append(np.where(gid == g)[0])
    return groups, members


def _map_codes(codes: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=int)
    out = np.full(codes.shape, -1, dtype=int)
    valid = (codes > 0) & (codes < mapping.shape[0])
    out[valid] = mapping[codes[valid]]
    return out


def _infer_household_sizes(individual_group_id: np.ndarray, n_households: int) -> np.ndarray:
    return np.bincount(individual_group_id, minlength=n_households + 1)[1:].astype(int)


def extract_pair_features_by_size(
    individual_data: np.ndarray,
    individual_group_id: np.ndarray,
    age_col: int,
    employ_col: int,
    age_code_to_order: np.ndarray | None = None,
    employ_code_to_group: np.ndarray | None = None,
    age_diff_n_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (age-diff, employ-diff) features for households of size >= 2.

    Returns
    -------
    features:
        Shape (n_households_with_features, 2), values are 1-based bins.
    group_ids:
        Household ids corresponding to each feature row.
    sizes:
        Household size for each feature row.
    invalid_group_ids:
        Household ids skipped due to invalid or missing age/employ codes.

    Parameters
    ----------
    age_diff_n_bins:
        Number of coarser age-difference bins (default 5).
        Reduces the raw 18-ordinal-position range to fewer, better-populated bins.
        Bin mapping: diff in [0..AGE_DIFF_MAX] → bin in [1..age_diff_n_bins] uniformly.
        Example (n=5): [0-3]→1, [4-7]→2, [8-10]→3, [11-14]→4, [15-17]→5.
        Feature space becomes age_diff_n_bins × 6 (e.g. 30 cells) instead of 18 × 6 = 108.
        Set to 18 to disable binning (original behaviour).
    """
    ind = np.asarray(individual_data, dtype=int)
    gid = np.asarray(individual_group_id, dtype=int).ravel()
    groups, members = _group_members(gid)

    _age_lookup = AGE_CODE_TO_ORDER if age_code_to_order is None else age_code_to_order
    _emp_lookup = EMPLOY_CODE_TO_GROUP if employ_code_to_group is None else employ_code_to_group
    age_ord = _map_codes(ind[:, age_col], _age_lookup)
    employ_grp = _map_codes(ind[:, employ_col], _emp_lookup)

    feats: list[list[int]] = []
    feat_gid: list[int] = []
    feat_sizes: list[int] = []
    invalid_gid: list[int] = []

    for g, idx in zip(groups, members, strict=False):
        size = int(idx.size)
        if size < 2:
            continue

        first_idx = idx[0]
        first_age = int(age_ord[first_idx])
        first_emp = int(employ_grp[first_idx])
        if first_age < 0 or first_emp < 0:
            invalid_gid.append(int(g))
            continue

        other_idx = idx[1:]
        other_age = age_ord[other_idx]
        other_emp = employ_grp[other_idx]
        valid_mask = (other_age >= 0) & (other_emp >= 0)
        if not np.any(valid_mask):
            invalid_gid.append(int(g))
            continue

        # MEAN across all (first member, other member) pairs, rounded to nearest integer.
        # For size=2, MEAN == MAX, preserving the paper's behaviour exactly.
        age_diff = round(float(np.mean(np.abs(other_age[valid_mask] - first_age))))
        emp_diff = round(float(np.mean(np.abs(other_emp[valid_mask] - first_emp))))

        # Coarser age bins: map raw diff [0..AGE_DIFF_MAX] → [1..age_diff_n_bins].
        # Reduces the 18-cell age axis to fewer, better-populated bins (default 5).
        # Formula: bin = floor(diff * n_bins / (AGE_DIFF_MAX + 1)) + 1, capped at n_bins.
        age_bin = min(int(age_diff) * age_diff_n_bins // (AGE_DIFF_MAX + 1) + 1, age_diff_n_bins)
        emp_bin = int(emp_diff) + 1   # employment groups already coarse (6 possible values)

        feats.append([age_bin, emp_bin])
        feat_gid.append(int(g))
        feat_sizes.append(size)

    if not feats:
        return (
            np.zeros((0, 2), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )

    return (
        np.asarray(feats, dtype=int),
        np.asarray(feat_gid, dtype=int),
        np.asarray(feat_sizes, dtype=int),
        np.asarray(invalid_gid, dtype=int),
    )


# paper_ref: Section 4.3, Eq. (3) and Eq. (2)
# matlab_ref: run_create_2.m:47-63, run_create_2.m:152-199
# intent: sample household attributes first, then individuals conditional on household class.
def generate_synthetic_population(
    n_households: int,
    pi_g: np.ndarray,
    pi_m: np.ndarray,
    phi_g: Sequence[np.ndarray],
    phi_m: Sequence[np.ndarray],
    household_size_col: int = 3,
    rng: np.random.Generator | None = None,
) -> SynthesisResult:
    """Generate synthetic households and individuals from model artifacts.

    Parameters
    ----------
    n_households:
        Number of households to generate.
    pi_g:
        Shape (G,), household class prior probabilities.
    pi_m:
        Shape (G, M), conditional individual class probabilities by household class.
    phi_g:
        List of household attribute conditional probability matrices.
        Each entry has shape (d_k, G).
    phi_m:
        List of individual attribute conditional probability matrices.
        Universal-class variant: each entry has shape (d_k, M).
    household_size_col:
        0-based column index for household size attribute in generated household_data.
    """
    rng = rng or np.random.default_rng() # allow deterministic sampling with provided Generator

    pi_g = np.asarray(pi_g, dtype=float)
    pi_g = pi_g / pi_g.sum()

    pi_m = np.asarray(pi_m, dtype=float) 
    if pi_m.ndim != 2:
        raise ValueError("pi_m must have shape (G, M)")

    # sample household classes for all households at once, based on pi_g.
    # classes for n_households, 0-based
    g_classes = rng.choice(np.arange(pi_g.size), size=n_households, p=pi_g)

    h_attrs = len(phi_g)
    household_data = np.zeros((n_households, h_attrs), dtype=int)

    for i in range(n_households):
        g = g_classes[i]
        for att, prob in enumerate(phi_g):
            p = np.asarray(prob, dtype=float)[:, g]
            p = p / p.sum() # probabilities for this attribute conditioned on household class g
            # +1 because categories are 1-based; categories are assumed to be contiguous and
            # start from 1, consistent with EM input/output conventions.
            household_data[i, att] = int(rng.choice(np.arange(p.size), p=p)) + 1

    # The household-size attribute is categorical; its code is used directly as the
    # number of members to generate for each household.
    sizes = household_data[:, household_size_col].astype(int)
    sizes = np.maximum(sizes, 1)
    n_individuals = int(np.sum(sizes))

    i_attrs = len(phi_m)
    individual_data = np.zeros((n_individuals, i_attrs), dtype=int)
    individual_group_id = np.zeros(n_individuals, dtype=int)
    individual_class = np.zeros(n_individuals, dtype=int)

    cursor = 0
    for i in range(n_households):
        g = g_classes[i]
        w = np.asarray(pi_m[g, :], dtype=float)
        w = w / w.sum()
        for _ in range(sizes[i]):
            m = int(rng.choice(np.arange(w.size), p=w))
            for att, prob in enumerate(phi_m):
                p = np.asarray(prob, dtype=float)[:, m]
                p = p / p.sum()
                individual_data[cursor, att] = int(rng.choice(np.arange(p.size), p=p)) + 1
            individual_group_id[cursor] = i + 1  # MATLAB-compatible 1-based household id.
            individual_class[cursor] = m + 1
            cursor += 1

    return SynthesisResult(
        household_data=household_data,
        individual_data=individual_data,
        individual_group_id=individual_group_id,
        household_class=g_classes + 1,
        individual_class=individual_class,
    )


# paper_ref: Section 4.4, Eq. (16) and rejection-sampling paragraph
# matlab_ref: run_two_within_the_same_4.m:112-120
# intent: apply acceptance-rejection correction per household size.
def rejection_filter_by_size(
    proposal_pairs: np.ndarray,
    target_prob: np.ndarray,
    proposal_prob: np.ndarray,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return acceptance mask and stats for a batch of households of one size.

    proposal_pairs columns:
    - col 0: transformed age difference bin index (1-based)
    - col 1: transformed employ difference bin index (1-based)
    """
    rng = rng or np.random.default_rng()

    if proposal_pairs.ndim != 2 or proposal_pairs.shape[1] != 2:
        raise ValueError("proposal_pairs must be shaped (n, 2)")

    if proposal_pairs.size == 0:
        return np.zeros((0,), dtype=bool), {
            "n_pairs": 0,
            "accept_rate": 0.0,
            "M_raw": 0.0,
            "M": 0.0,
            "f1_zero_bins": 0,
        }

    y1 = proposal_pairs[:, 0].astype(int) - 1
    y2 = proposal_pairs[:, 1].astype(int) - 1

    if np.any(y1 < 0) or np.any(y2 < 0):
        raise ValueError("proposal_pairs must be 1-based indices")

    eps = 1e-15
    # M_raw computed only over bins where BOTH distributions are strictly positive.
    # If we used eps in the denominator for all bins, any zero-proposal / non-zero-target
    # cell would produce a ratio of ~f_target/1e-15, making M_raw meaningless and
    # always triggering the 3.0 cap.
    # Computing over the overlap gives a meaningful M_raw.
    both_positive = (proposal_prob > eps) & (target_prob > 0)
    if both_positive.any():
        M_raw = float(np.max(target_prob[both_positive] / proposal_prob[both_positive]))
    else:
        M_raw = 0.0
    M = min(M_raw, 3.0) if M_raw > 0 else 1.0

    # Acceptance ratio: eps in denominator prevents divide-by-zero for zero-proposal bins.
    # Those bins get alpha >> 1 → clipped to 1 (always accept if target has mass there).
    f1_zero_bins = int(np.sum((proposal_prob <= eps) & (target_prob > 0)))
    alpha = target_prob[y1, y2] / (M * np.maximum(proposal_prob[y1, y2], eps))
    alpha = np.clip(alpha, 0.0, 1.0)

    if verbose and alpha.size > 0:
        accept_ready = int(np.sum(alpha > 0))
        cap_note = f" -> capped={M:.3f}" if M_raw > 3.0 else ""
        print(f"      [ALPHA] Scaling factor M: raw={M_raw:.3f}{cap_note} -> applied={M:.3f}")
        # "Households" = number of synthetic households of this size being evaluated.
        # This is NOT the number of feature bins (age_bins × emp_bins).
        print(
            f"      [ALPHA] Households: {alpha.size}, accept-ready (alpha>0): {accept_ready} "
            f"({100*accept_ready/alpha.size:.1f}%)"
        )

    u = rng.random(alpha.shape[0])
    accept = u <= alpha
    stats = {
        "n_pairs": int(alpha.size),
        "accept_rate": float(np.mean(accept)) if alpha.size > 0 else 0.0,
        "M_raw": float(M_raw),
        "M": float(M),
        "f1_zero_bins": f1_zero_bins,
    }
    return accept, stats


def rejection_filter_two_person_households(
    proposal_pairs: np.ndarray,
    target_prob: np.ndarray,
    proposal_prob: np.ndarray,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> np.ndarray:
    warnings.warn(
        "rejection_filter_two_person_households is deprecated; "
        "use rejection_filter_by_size instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    accept, _ = rejection_filter_by_size(
        proposal_pairs=proposal_pairs,
        target_prob=target_prob,
        proposal_prob=proposal_prob,
        rng=rng,
        verbose=verbose,
    )
    return accept


def two_person_pair_features(
    individual_data: np.ndarray,
    individual_group_id: np.ndarray,
    age_col: int,
    sex_col: int,
) -> tuple[np.ndarray, np.ndarray]:
    warnings.warn(
        "two_person_pair_features is deprecated; use extract_pair_features_by_size instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    feat, gid, sizes, _ = extract_pair_features_by_size(
        individual_data=individual_data,
        individual_group_id=individual_group_id,
        age_col=age_col,
        employ_col=sex_col,
    )
    mask = sizes == 2
    return feat[mask], gid[mask]


def _rejection_diagnostics_for_batch(
    stage: str,
    batch_num: int,
    before_household_data: np.ndarray,
    after_household_data: np.ndarray,
    stats_by_size: dict[int, dict[str, float | int | bool]],
    household_size_col: int = 3,
) -> None:
    """Print diagnostics for rejection sampling across household sizes."""
    print(f"\n{'='*100}")
    print(f"[DIAG-REJECT] {stage} {batch_num}: REJECTION SAMPLING VERIFICATION")
    print(f"{'='*100}")

    before_sizes = before_household_data[:, household_size_col].astype(int)
    after_sizes = after_household_data[:, household_size_col].astype(int)

    before_counts: dict[int, int] = {}
    after_counts: dict[int, int] = {}
    for s in np.unique(np.concatenate([before_sizes, after_sizes])):
        before_counts[int(s)] = int(np.sum(before_sizes == s))
        after_counts[int(s)] = int(np.sum(after_sizes == s))

    print("\n1. HOUSEHOLD SIZE DISTRIBUTION (before vs after rejection):")
    print(f"{'Size':>6} {'Before':>12} {'After':>12} {'Dropped':>12} {'Drop%':>10}")
    print("-" * 70)

    for size in sorted(set(before_counts.keys()) | set(after_counts.keys())):
        before = before_counts.get(size, 0)
        after = after_counts.get(size, 0)
        dropped = max(before - after, 0)
        drop_pct = 100.0 * dropped / before if before > 0 else 0.0
        print(f"{size:6d} {before:12d} {after:12d} {dropped:12d} {drop_pct:9.1f}%")

    print("\n2. PER-SIZE ACCEPTANCE SUMMARY:")
    print(
        f"{'Size':>6} {'Proposed':>12} {'Accepted':>12} {'Accept%':>10} "
        f"{'M':>6} {'f1=0':>8} {'Target?':>9}"
    )
    print("-" * 80)
    for size in sorted(stats_by_size.keys()):
        stats = stats_by_size[size]
        n_proposed = int(stats.get("n_proposed", 0))
        n_accepted = int(stats.get("n_accepted", 0))
        accept_rate = float(stats.get("accept_rate", 0.0)) * 100.0
        M = float(stats.get("M", 0.0))
        f1_zero = int(stats.get("f1_zero_bins", 0))
        has_target = "yes" if not bool(stats.get("target_missing", False)) else "no"
        print(
            f"{int(size):6d} {n_proposed:12d} {n_accepted:12d} {accept_rate:9.1f}% "
            f"{M:6.2f} {f1_zero:8d} {has_target:>9}"
        )

    print(f"\n{'='*100}\n")


def _feature_distribution(features: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Build an empirical 2D distribution from 1-based feature bins.

    Notes
    -----
    `features` follows the MATLAB-style 1-based bins (e.g., y1=1,2,...).
    The output array is NumPy (0-based), so counts are written to
    ``out[y1 - 1, y2 - 1]``.
    """
    # If there are no observations, return an all-zero table with a safe shape.
    if features.size == 0:
        if shape is None:
            return np.zeros((1, 1), dtype=float)
        return np.zeros(shape, dtype=float)

    # Infer the contingency-table size from observed max 1-based bin ids.
    if shape is None:
        shape = (int(np.max(features[:, 0])), int(np.max(features[:, 1])))

    # Accumulate counts into a 2D table.
    out = np.zeros(shape, dtype=float)
    for y1, y2 in features:
        # Skip malformed/out-of-range bins rather than raising.
        if y1 < 1 or y2 < 1 or y1 > shape[0] or y2 > shape[1]:
            continue
        # Map 1-based bins (paper/MATLAB convention) to 0-based NumPy indices.
        out[y1 - 1, y2 - 1] += 1.0

    # Normalize counts into probabilities; keep all-zero table if no valid counts.
    total = out.sum()
    if total <= 0:
        return out
    return out / total


# paper_ref: Section 4.4, Eq. (16) and rejection-sampling step
# matlab_ref: run_two_within_the_same_4.m:112-120 and 175-241
# intent: postprocess synthetic population to correct household relations by size.
def apply_rejection_by_size(
    synthetic: SynthesisResult,
    target_individual_data: np.ndarray,
    target_individual_group_id: np.ndarray,
    age_col: int = 0,
    employ_col: int = 1,
    household_size_col: int | None = 3,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    stage: str = "Rejection",
    batch_num: int = 0,
    age_code_to_order: np.ndarray | None = None,
    employ_code_to_group: np.ndarray | None = None,
    min_target_samples: int = 30,
    age_diff_n_bins: int = 5,
) -> SynthesisResult:
    """Apply rejection-sampling correction for all household sizes >= 2."""
    rng = rng or np.random.default_rng()

    prop_feat, prop_gid, prop_sizes, prop_invalid = extract_pair_features_by_size(
        synthetic.individual_data,
        synthetic.individual_group_id,
        age_col=age_col,
        employ_col=employ_col,
        age_code_to_order=age_code_to_order,
        employ_code_to_group=employ_code_to_group,
        age_diff_n_bins=age_diff_n_bins,
    )
    tgt_feat, _, tgt_sizes, _ = extract_pair_features_by_size(
        np.asarray(target_individual_data, dtype=int),
        np.asarray(target_individual_group_id, dtype=int),
        age_col=age_col,
        employ_col=employ_col,
        age_code_to_order=age_code_to_order,
        employ_code_to_group=employ_code_to_group,
        age_diff_n_bins=age_diff_n_bins,
    )

    if prop_feat.size == 0 or tgt_feat.size == 0:
        if verbose:
            print("No household features available for rejection correction; skipping.")
        return synthetic

    stats_by_size: dict[int, dict[str, float | int | bool]] = {}
    accept_mask = np.zeros((prop_feat.shape[0],), dtype=bool)

    # Upfront sparsity-guard report: identify sizes that will be bypassed before the loop.
    sparsity_bypassed = sorted([
        int(s) for s in set(prop_sizes.tolist())
        if int(np.sum(tgt_sizes == s)) < min_target_samples
    ])
    if sparsity_bypassed:
        print(
            f"[INFO-{stage} {batch_num}] Sparsity guard (< {min_target_samples} target "
            f"examples): sizes {sparsity_bypassed} — rejection bypassed, all accepted."
        )

    for size in sorted(set(prop_sizes.tolist())):
        prop_mask = prop_sizes == size
        tgt_mask = tgt_sizes == size

        n_proposed = int(np.sum(prop_mask))
        if n_proposed == 0:
            continue

        n_target = int(np.sum(tgt_mask))
        if n_target < min_target_samples:
            # Too few target examples: noisy distribution is worse than no correction.
            # Always accept households of this size (same behaviour as target_missing).
            accept_mask[prop_mask] = True
            stats_by_size[int(size)] = {
                "n_proposed": n_proposed,
                "n_accepted": n_proposed,
                "accept_rate": 1.0,
                "M_raw": 0.0,
                "M": 0.0,
                "f1_zero_bins": 0,
                "target_missing": True,
            }
            continue

        shape = (
            max(int(np.max(prop_feat[prop_mask, 0])), int(np.max(tgt_feat[tgt_mask, 0]))),
            max(int(np.max(prop_feat[prop_mask, 1])), int(np.max(tgt_feat[tgt_mask, 1]))),
        )

        proposal_prob = _feature_distribution(prop_feat[prop_mask], shape=shape)
        target_prob = _feature_distribution(tgt_feat[tgt_mask], shape=shape)

        if verbose:
            print(
                f"\n[REJECT-{stage} {batch_num}] Applying rejection for size={size} households..."
            )

        accept_size, stats = rejection_filter_by_size(
            prop_feat[prop_mask],
            target_prob,
            proposal_prob,
            rng=rng,
            verbose=verbose,
        )
        accept_mask[prop_mask] = accept_size

        stats_by_size[int(size)] = {
            "n_proposed": n_proposed,
            "n_accepted": int(np.sum(accept_size)),
            "accept_rate": float(np.mean(accept_size)) if accept_size.size > 0 else 0.0,
            "M_raw": stats.get("M_raw", 0.0),
            "M": stats.get("M", 0.0),
            "f1_zero_bins": int(stats.get("f1_zero_bins", 0)),
            "target_missing": False,
        }

    keep_groups = set(prop_gid[accept_mask].tolist())
    keep_groups.update(prop_invalid.tolist())

    all_groups = np.arange(1, synthetic.household_data.shape[0] + 1)
    n_hh_cols = synthetic.household_data.shape[1]
    if household_size_col is not None and 0 <= household_size_col < n_hh_cols:
        sizes = synthetic.household_data[:, household_size_col].astype(int)
    else:
        sizes = _infer_household_sizes(
            synthetic.individual_group_id, synthetic.household_data.shape[0]
        )

    keep_list: list[int] = []
    for g in all_groups:
        if int(sizes[g - 1]) < 2:
            keep_list.append(int(g))
        elif int(g) in keep_groups:
            keep_list.append(int(g))

    keep_groups_arr = np.asarray(keep_list, dtype=int)
    if keep_groups_arr.size == 0:
        return synthetic

    if verbose:
        after_mask = np.isin(all_groups, keep_groups_arr)
        after_household = synthetic.household_data[after_mask, :]
        _rejection_diagnostics_for_batch(
            stage=stage,
            batch_num=batch_num,
            before_household_data=synthetic.household_data,
            after_household_data=after_household,
            stats_by_size=stats_by_size,
            household_size_col=household_size_col or 3,
        )

    old_to_new = {int(g): i + 1 for i, g in enumerate(keep_groups_arr.tolist())}

    keep_house_mask = np.isin(all_groups, keep_groups_arr)
    new_household_data = synthetic.household_data[keep_house_mask, :]
    new_household_class = synthetic.household_class[keep_house_mask]

    keep_ind_mask = np.isin(synthetic.individual_group_id, keep_groups_arr)
    new_ind_data = synthetic.individual_data[keep_ind_mask, :]
    new_ind_class = synthetic.individual_class[keep_ind_mask]
    new_ind_gid_old = synthetic.individual_group_id[keep_ind_mask]
    new_ind_gid = np.array([old_to_new[int(g)] for g in new_ind_gid_old], dtype=int)

    return SynthesisResult(
        household_data=new_household_data,
        individual_data=new_ind_data,
        individual_group_id=new_ind_gid,
        household_class=new_household_class,
        individual_class=new_ind_class,
    )


def apply_two_person_rejection(
    synthetic: SynthesisResult,
    target_individual_data: np.ndarray,
    target_individual_group_id: np.ndarray,
    age_col: int = 0,
    sex_col: int = 1,
    household_size_col: int | None = 3,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    stage: str = "Rejection",
    batch_num: int = 0,
) -> SynthesisResult:
    warnings.warn(
        "apply_two_person_rejection is deprecated; use apply_rejection_by_size instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return apply_rejection_by_size(
        synthetic=synthetic,
        target_individual_data=target_individual_data,
        target_individual_group_id=target_individual_group_id,
        age_col=age_col,
        employ_col=sex_col,
        household_size_col=household_size_col,
        rng=rng,
        verbose=verbose,
        stage=stage,
        batch_num=batch_num,
    )


def _slice_first_households(synth: SynthesisResult, n_keep: int, hh_offset: int) -> SynthesisResult:
    """Take the first n_keep households from a SynthesisResult, re-indexing gids by hh_offset."""
    keep_h = min(int(n_keep), int(synth.household_data.shape[0]))
    if keep_h <= 0:
        return SynthesisResult(
            household_data=np.zeros((0, synth.household_data.shape[1]), dtype=int),
            individual_data=np.zeros((0, synth.individual_data.shape[1]), dtype=int),
            individual_group_id=np.zeros((0,), dtype=int),
            household_class=np.zeros((0,), dtype=int),
            individual_class=np.zeros((0,), dtype=int),
        )
    hh = synth.household_data[:keep_h, :]
    hh_class = synth.household_class[:keep_h]
    ind_mask = synth.individual_group_id <= keep_h
    return SynthesisResult(
        household_data=hh,
        individual_data=synth.individual_data[ind_mask, :],
        individual_group_id=synth.individual_group_id[ind_mask] + int(hh_offset),
        household_class=hh_class,
        individual_class=synth.individual_class[ind_mask],
    )


def _slice_households_by_indices(
    synth: SynthesisResult,
    keep_indices: np.ndarray,
    hh_offset: int,
) -> SynthesisResult:
    if keep_indices.size == 0:
        return SynthesisResult(
            household_data=np.zeros((0, synth.household_data.shape[1]), dtype=int),
            individual_data=np.zeros((0, synth.individual_data.shape[1]), dtype=int),
            individual_group_id=np.zeros((0,), dtype=int),
            household_class=np.zeros((0,), dtype=int),
            individual_class=np.zeros((0,), dtype=int),
        )

    keep_indices = np.asarray(np.sort(keep_indices), dtype=int)
    keep_groups = keep_indices + 1
    old_to_new = {int(g): i + 1 for i, g in enumerate(keep_groups.tolist())}

    keep_house_mask = np.zeros((synth.household_data.shape[0],), dtype=bool)
    keep_house_mask[keep_indices] = True
    new_household_data = synth.household_data[keep_house_mask, :]
    new_household_class = synth.household_class[keep_house_mask]

    keep_ind_mask = np.isin(synth.individual_group_id, keep_groups)
    new_ind_data = synth.individual_data[keep_ind_mask, :]
    new_ind_class = synth.individual_class[keep_ind_mask]
    new_ind_gid_old = synth.individual_group_id[keep_ind_mask]
    new_ind_gid = np.array([old_to_new[int(g)] for g in new_ind_gid_old], dtype=int)

    return SynthesisResult(
        household_data=new_household_data,
        individual_data=new_ind_data,
        individual_group_id=new_ind_gid + int(hh_offset),
        household_class=new_household_class,
        individual_class=new_ind_class,
    )


def _size_counts_from_households(
    household_data: np.ndarray,
    household_size_col: int,
) -> dict[int, int]:
    sizes = household_data[:, household_size_col].astype(int)
    counts: dict[int, int] = {}
    for size in sizes:
        counts[int(size)] = counts.get(int(size), 0) + 1
    return counts


def _scale_size_counts(counts: dict[int, int], target_total: int) -> dict[int, int]:
    if target_total <= 0:
        return {size: 0 for size in counts}

    raw_total = int(sum(counts.values()))
    if raw_total <= 0:
        return {size: 0 for size in counts}

    expected = {size: counts[size] * target_total / raw_total for size in counts}
    base = {size: int(math.floor(val)) for size, val in expected.items()}
    remainder = target_total - sum(base.values())
    if remainder > 0:
        frac_order = sorted(
            expected.items(), key=lambda kv: kv[1] - math.floor(kv[1]), reverse=True
        )
        for size, _ in frac_order[:remainder]:
            base[size] += 1

    return base


# Take up to the needed count of households by size from the given SynthesisResult, returning
# the taken subset and the updated need_by_size counts.
def _take_households_by_size(
    synth: SynthesisResult,
    need_by_size: dict[int, int],
    household_size_col: int,
    hh_offset: int,
) -> tuple[SynthesisResult, dict[int, int]]:
    sizes = synth.household_data[:, household_size_col].astype(int)
    keep_indices: list[int] = []
    for idx, size in enumerate(sizes):
        need = need_by_size.get(int(size), 0)
        if need > 0:
            keep_indices.append(idx)
            need_by_size[int(size)] = need - 1

    taken = _slice_households_by_indices(
        synth=synth,
        keep_indices=np.asarray(keep_indices, dtype=int),
        hh_offset=hh_offset,
    )
    return taken, need_by_size


# paper_ref: Section 4.3 + Section 4.4 (Eq. 16) combined with target-count guarantee
# matlab_ref: run_create_2.m + run_two_within_the_same_4.m (oversample-then-reject pattern)
# intent: produce exactly n_households after rejection by redrawing fresh batches
#         instead of MATLAB's row-duplication hack, so the output has no duplicate rows.
def generate_with_multi_size_rejection(
    n_households: int,
    pi_g: np.ndarray,
    pi_m: np.ndarray,
    phi_g: Sequence[np.ndarray],
    phi_m: Sequence[np.ndarray],
    target_individual_data: np.ndarray,
    target_individual_group_id: np.ndarray,
    *,
    household_size_col: int = 3,
    age_col: int = 0,
    employ_col: int = 1,
    initial_oversample: float = 1.5,
    topup_factor: float = 1.7,
    max_topup_iters: int = 10,
    rng: np.random.Generator | None = None,
    age_code_to_order: np.ndarray | None = None,
    employ_code_to_group: np.ndarray | None = None,
    min_target_samples: int = 30,
    age_diff_n_bins: int = 5,
) -> SynthesisResult:
    """Generate exactly n_households synthetic households with size-aware rejection.

    The target size counts are computed from the initial synthetic draw and scaled
    to ``n_households`` so the final output preserves the model-implied size mix.
    """
    if n_households <= 0:
        raise ValueError("n_households must be positive")
    if initial_oversample < 1.0:
        raise ValueError("initial_oversample must be >= 1.0")
    if topup_factor < 1.0:
        raise ValueError("topup_factor must be >= 1.0")
    if max_topup_iters < 0:
        raise ValueError("max_topup_iters must be non-negative")

    rng = rng or np.random.default_rng()

    first_n = max(int(math.ceil(n_households * initial_oversample)), n_households)
    raw = generate_synthetic_population(
        n_households=first_n,
        pi_g=pi_g,
        pi_m=pi_m,
        phi_g=phi_g,
        phi_m=phi_m,
        household_size_col=household_size_col,
        rng=rng,
    )

    if household_size_col < 0 or household_size_col >= raw.household_data.shape[1]:
        raise ValueError("household_size_col out of range for household_data")

    raw_counts = _size_counts_from_households(raw.household_data, household_size_col)
    target_counts = _scale_size_counts(raw_counts, n_households)

    kept = apply_rejection_by_size(
        synthetic=raw,
        target_individual_data=target_individual_data,
        target_individual_group_id=target_individual_group_id,
        age_col=age_col,
        employ_col=employ_col,
        household_size_col=household_size_col,
        rng=rng,
        verbose=True,
        stage="First-pass",
        batch_num=0,
        age_code_to_order=age_code_to_order,
        employ_code_to_group=employ_code_to_group,
        min_target_samples=min_target_samples,
        age_diff_n_bins=age_diff_n_bins,
    )

    parts: list[SynthesisResult] = []
    hh_offset = 0
    take, target_counts = _take_households_by_size(
        synth=kept,
        need_by_size=target_counts,
        household_size_col=household_size_col,
        hh_offset=hh_offset,
    )
    parts.append(take)
    hh_offset += take.household_data.shape[0]
    remaining = int(sum(target_counts.values()))

    iters = 0
    while remaining > 0:
        if iters >= max_topup_iters:
            raise RuntimeError(
                f"generate_with_multi_size_rejection: could not reach target "
                f"{n_households} households after {max_topup_iters} top-up iterations "
                f"(short by {remaining}). Try increasing max_topup_iters or topup_factor."
            )
        iters += 1
        batch_n = max(int(math.ceil(remaining * topup_factor)), 200)
        batch_raw = generate_synthetic_population(
            n_households=batch_n,
            pi_g=pi_g,
            pi_m=pi_m,
            phi_g=phi_g,
            phi_m=phi_m,
            household_size_col=household_size_col,
            rng=rng,
        )
        batch_kept = apply_rejection_by_size(
            synthetic=batch_raw,
            target_individual_data=target_individual_data,
            target_individual_group_id=target_individual_group_id,
            age_col=age_col,
            employ_col=employ_col,
            household_size_col=household_size_col,
            rng=rng,
            verbose=True,
            stage="Top-up batch",
            batch_num=iters,
            age_code_to_order=age_code_to_order,
            employ_code_to_group=employ_code_to_group,
            min_target_samples=min_target_samples,
            age_diff_n_bins=age_diff_n_bins,
        )
        batch_take, target_counts = _take_households_by_size(
            synth=batch_kept,
            need_by_size=target_counts,
            household_size_col=household_size_col,
            hh_offset=hh_offset,
        )
        if batch_take.household_data.shape[0] == 0:
            remaining = int(sum(target_counts.values()))
            continue
        parts.append(batch_take)
        hh_offset += batch_take.household_data.shape[0]
        remaining = int(sum(target_counts.values()))

    household_data = np.vstack([p.household_data for p in parts])
    household_class = np.concatenate([p.household_class for p in parts])
    individual_data = np.vstack([p.individual_data for p in parts])
    individual_group_id = np.concatenate([p.individual_group_id for p in parts])
    individual_class = np.concatenate([p.individual_class for p in parts])

    return SynthesisResult(
        household_data=household_data,
        individual_data=individual_data,
        individual_group_id=individual_group_id,
        household_class=household_class,
        individual_class=individual_class,
    )


def generate_with_two_person_rejection(
    n_households: int,
    pi_g: np.ndarray,
    pi_m: np.ndarray,
    phi_g: Sequence[np.ndarray],
    phi_m: Sequence[np.ndarray],
    target_individual_data: np.ndarray,
    target_individual_group_id: np.ndarray,
    *,
    household_size_col: int = 3,
    age_col: int = 0,
    sex_col: int = 1,
    initial_oversample: float = 1.5,
    topup_factor: float = 1.7,
    max_topup_iters: int = 20,
    rng: np.random.Generator | None = None,
) -> SynthesisResult:
    warnings.warn(
        "generate_with_two_person_rejection is deprecated; "
        "use generate_with_multi_size_rejection instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return generate_with_multi_size_rejection(
        n_households=n_households,
        pi_g=pi_g,
        pi_m=pi_m,
        phi_g=phi_g,
        phi_m=phi_m,
        target_individual_data=target_individual_data,
        target_individual_group_id=target_individual_group_id,
        household_size_col=household_size_col,
        age_col=age_col,
        employ_col=sex_col,
        initial_oversample=initial_oversample,
        topup_factor=topup_factor,
        max_topup_iters=max_topup_iters,
        rng=rng,
    )
