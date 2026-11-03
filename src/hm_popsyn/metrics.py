"""SRMSE and precision, recall and F1 for comparing a synthetic population with a target.

SRMSE compares marginal and bivariate distributions (Borysov, Rich & Pereira, 2019).
Precision, recall and F1 compare complete attribute combinations (Kim & Bansal, 2023):
precision is the share of synthetic rows whose combination exists in the target, recall the
share of target rows whose combination appears in the synthetic data.

The functions take pandas DataFrames with one categorical attribute per column.
``member_table`` builds such a table from the arrays this package uses, one row per individual
with the household attributes repeated for every member.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def srmse(
    pop_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    category_sets: dict[str, list] | None = None,
) -> dict[str, float]:
    """Standardized RMSE over marginal and bivariate distributions.

    For K variables the marginal vector concatenates K distributions (each summing to 1).
    The bivariate vector concatenates K*(K-1)/2 joint distributions (each summing to 1).
    SRMSE = RMSE / mean(target proportions). Lower is better.

    category_sets: optional {column: sorted list of codes} that fixes each column's bins,
        for example to every code in the full target, rather than the union of codes
        observed in the two tables. With N observed bins and B extra empty bins, SRMSE
        grows by a factor sqrt((N + B) / N), so fix the bins when comparing runs.
    """
    common_cols = sorted(set(pop_df.columns) & set(gen_df.columns))
    pop = pop_df[common_cols]
    gen = gen_df[common_cols]

    # Marginal
    pop_props, gen_props = [], []
    for col in common_cols:
        pop_counts = pop[col].dropna().value_counts()
        gen_counts = gen[col].dropna().value_counts()
        if category_sets is not None and col in category_sets:
            all_vals = category_sets[col]
        else:
            all_vals = sorted(set(pop_counts.index) | set(gen_counts.index))
        p = np.array([pop_counts.get(v, 0) for v in all_vals], dtype=float)
        g = np.array([gen_counts.get(v, 0) for v in all_vals], dtype=float)
        if p.sum() > 0:
            p /= p.sum()
        if g.sum() > 0:
            g /= g.sum()
        pop_props.append(p)
        gen_props.append(g)

    pop_props_arr = np.concatenate(pop_props)
    gen_props_arr = np.concatenate(gen_props)
    mean_pop = pop_props_arr.mean()
    srmse_mar = (
        float(np.sqrt(np.mean((pop_props_arr - gen_props_arr) ** 2)) / mean_pop)
        if mean_pop > 0
        else float("nan")
    )

    # Bivariate
    pop_bi, gen_bi = [], []
    for col1, col2 in combinations(common_cols, 2):
        pop_pair = pop[[col1, col2]].dropna()
        gen_pair = gen[[col1, col2]].dropna()
        if len(pop_pair) == 0:
            continue
        pop_ct = pd.crosstab(pop_pair[col1], pop_pair[col2])
        gen_ct = pd.crosstab(gen_pair[col1], gen_pair[col2])
        if category_sets is not None:
            row_cats = category_sets.get(col1, sorted(set(pop_ct.index) | set(gen_ct.index)))
            col_cats = category_sets.get(col2, sorted(set(pop_ct.columns) | set(gen_ct.columns)))
        else:
            row_cats = sorted(set(pop_ct.index) | set(gen_ct.index))
            col_cats = sorted(set(pop_ct.columns) | set(gen_ct.columns))
        pop_ct = pop_ct.reindex(index=row_cats, columns=col_cats, fill_value=0)
        gen_ct = gen_ct.reindex(index=row_cats, columns=col_cats, fill_value=0)
        p = pop_ct.values.flatten().astype(float)
        g = gen_ct.values.flatten().astype(float)
        if p.sum() > 0:
            p /= p.sum()
        if g.sum() > 0:
            g /= g.sum()
        pop_bi.append(p)
        gen_bi.append(g)

    if pop_bi:
        pop_bi_arr = np.concatenate(pop_bi)
        gen_bi_arr = np.concatenate(gen_bi)
        mean_bi = pop_bi_arr.mean()
        srmse_bi = (
            float(np.sqrt(np.mean((pop_bi_arr - gen_bi_arr) ** 2)) / mean_bi)
            if mean_bi > 0
            else float("nan")
        )
    else:
        srmse_bi = float("nan")

    return {"marginal": srmse_mar, "bivariate": srmse_bi}


def precision_recall_f1(pop_df: pd.DataFrame, gen_df: pd.DataFrame) -> dict[str, float | int]:
    """Precision, recall and F1 by exact matching of complete attribute combinations.

    A synthetic row counts as a match if the same combination exists anywhere in the target.
    Precision = matches among synthetic rows / number of synthetic rows.
    Recall = target rows whose combination is generated / number of target rows.
    """
    common_cols = sorted(set(pop_df.columns) & set(gen_df.columns))
    pop_clean = pop_df[common_cols].dropna()
    gen_clean = gen_df[common_cols].dropna()

    pop_tuples = pop_clean.apply(tuple, axis=1)
    gen_tuples = gen_clean.apply(tuple, axis=1)

    pop_set = set(pop_tuples)
    gen_set = set(gen_tuples)

    precision = (
        sum(1 for t in gen_tuples if t in pop_set) / len(gen_tuples) if len(gen_tuples) > 0 else 0.0
    )
    recall = (
        sum(1 for t in pop_tuples if t in gen_set) / len(pop_tuples) if len(pop_tuples) > 0 else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "unique_pop": len(pop_set),
        "unique_gen": len(gen_set),
        "matching_types": len(pop_set & gen_set),
    }


def member_table(
    household_data: np.ndarray,
    individual_data: np.ndarray,
    individual_group_id: np.ndarray,
    household_names: list[str] | None = None,
    individual_names: list[str] | None = None,
) -> pd.DataFrame:
    """One row per individual: household attributes repeated, then individual attributes.

    ``individual_group_id`` holds 1-based household ids, as in ``PreparedData.indgid`` and
    ``SynthesisResult.individual_group_id``. Default column names are ``hh_1``, ``hh_2``, ...
    and ``ind_1``, ``ind_2``, ...
    """
    hh = np.asarray(household_data)
    ind = np.asarray(individual_data)
    gid = np.asarray(individual_group_id, dtype=int).ravel()
    if hh.ndim != 2 or ind.ndim != 2:
        raise ValueError("household_data and individual_data must be 2D")
    if ind.shape[0] != gid.shape[0]:
        raise ValueError("individual_data and individual_group_id must have the same length")
    if gid.size and (gid.min() < 1 or gid.max() > hh.shape[0]):
        raise ValueError("individual_group_id must be 1-based household ids within household_data")

    hh_names = household_names or [f"hh_{k + 1}" for k in range(hh.shape[1])]
    ind_names = individual_names or [f"ind_{k + 1}" for k in range(ind.shape[1])]
    if len(hh_names) != hh.shape[1] or len(ind_names) != ind.shape[1]:
        raise ValueError("column names must match the number of attributes")

    return pd.DataFrame(np.hstack([hh[gid - 1], ind]), columns=hh_names + ind_names)


def category_sets_from(reference: pd.DataFrame) -> dict[str, list]:
    """Sorted codes observed in each column of a reference table, for ``srmse``."""
    return {col: sorted(reference[col].dropna().unique().tolist()) for col in reference.columns}


def evaluate_synthetic(
    target: pd.DataFrame,
    synthetic: pd.DataFrame,
    label: str = "",
    category_sets: dict[str, list] | None = None,
    verbose: bool = True,
) -> dict:
    """SRMSE and precision, recall and F1 of a synthetic table against a target table.

    Columns present in both tables are compared. Bins are fixed to the codes found in
    ``target`` unless ``category_sets`` is given, so results stay comparable across runs.
    Pass the full target population here when you have it, not only the training sample.
    """
    common = sorted(set(target.columns) & set(synthetic.columns))
    if not common:
        raise ValueError("target and synthetic share no columns")
    cats = category_sets if category_sets is not None else category_sets_from(target[common])
    srmse_res = srmse(target[common], synthetic[common], category_sets=cats)
    prf_res = precision_recall_f1(target[common], synthetic[common])
    cat_counts = {
        col: len(cats.get(col, sorted(set(target[col].dropna()) | set(synthetic[col].dropna()))))
        for col in common
    }

    if verbose:
        width = 60
        print("=" * width)
        print(f"  {label}" if label else "  Synthetic vs target")
        print("-" * width)
        print(f"  {'Target rows:':<28}{len(target):>12,}")
        print(f"  {'Synthetic rows:':<28}{len(synthetic):>12,}")
        print(f"  {'Variables (K):':<28}{len(common):>12}")
        print(f"  {'Marginal bins:':<28}{sum(cat_counts.values()):>12,}")
        print(f"  {'SRMSE marginal:':<28}{srmse_res['marginal']:>12.4f}")
        print(f"  {'SRMSE bivariate:':<28}{srmse_res['bivariate']:>12.4f}")
        print(f"  {'Precision:':<28}{prf_res['precision']:>12.4f}")
        print(f"  {'Recall:':<28}{prf_res['recall']:>12.4f}")
        print(f"  {'F1:':<28}{prf_res['f1']:>12.4f}")
        print(f"  {'Distinct target / synthetic:':<28}"
              f"{prf_res['unique_pop']:>6,} / {prf_res['unique_gen']:,}")
        print("=" * width)

    return {"columns": common, "srmse": srmse_res, "prf": prf_res, "cat_counts": cat_counts}
