"""Input preparation utilities for hierarchical household-individual data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class PreparedData:
    """Prepared arrays for EM/synthesis pipeline."""

    group_ids: np.ndarray
    grpdata: np.ndarray
    indgid: np.ndarray
    inddata: np.ndarray


# matlab_ref: run_multilevel_1.m:8-27 and run_create_2.m:8-27
# notes: reproduces sorting and one-based category coding assumptions.
def prepare_from_merged_table(
    data: np.ndarray,
    household_id_col: int = 0,
    individual_id_col: int = 1,
    household_attr_cols: Sequence[int] = (2, 3, 4, 5),
    individual_attr_cols: Sequence[int] = (6, 7, 8, 10, 11),
) -> PreparedData:
    """Prepare arrays from merged person-level table used by MATLAB scripts."""
    arr = np.asarray(data, dtype=int)
    if arr.ndim != 2:
        raise ValueError("data must be 2D")

    order = np.lexsort((arr[:, individual_id_col], arr[:, household_id_col]))
    arr = arr[order, :]

    # Match MATLAB category remapping: shift so minimum is 1 globally.
    arr = arr - arr.min(axis=0, keepdims=True) + 1

    indgid = arr[:, household_id_col].astype(int)
    # Re-map to contiguous 1..N household ids to preserve accumarray semantics.
    unique_gid = np.unique(indgid)
    gid_map = {int(g): i + 1 for i, g in enumerate(unique_gid.tolist())}
    indgid = np.array([gid_map[int(g)] for g in indgid], dtype=int)

    grp_ids = np.arange(1, unique_gid.size + 1, dtype=int)

    # Extract one row per household for group-level attributes.
    first_idx = np.unique(indgid, return_index=True)[1]
    grpdata = arr[first_idx[:, None], np.asarray(household_attr_cols)[None, :]].reshape(
        first_idx.size, -1
    )

    inddata = arr[:, np.asarray(individual_attr_cols)]

    return PreparedData(
        group_ids=grp_ids,
        grpdata=grpdata.astype(int),
        indgid=indgid.astype(int),
        inddata=inddata.astype(int),
    )


def load_csv(path: str | Path, skip_header: bool = True) -> np.ndarray:
    """Load a CSV table into numpy.

    The loader is strict and expects numeric values.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return np.genfromtxt(p, delimiter=",", skip_header=1 if skip_header else 0)
