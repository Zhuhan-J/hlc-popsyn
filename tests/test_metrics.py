import math

import numpy as np
import pandas as pd
import pytest

from hm_popsyn.metrics import (
    category_sets_from,
    evaluate_synthetic,
    member_table,
    precision_recall_f1,
    srmse,
)


def _table():
    return pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": [1, 2, 1, 2, 2]})


def test_identical_tables_give_zero_srmse_and_perfect_prf():
    df = _table()
    s = srmse(df, df.copy())
    assert s["marginal"] == pytest.approx(0.0)
    assert s["bivariate"] == pytest.approx(0.0)
    prf = precision_recall_f1(df, df.copy())
    assert (prf["precision"], prf["recall"], prf["f1"]) == (1.0, 1.0, 1.0)
    assert prf["unique_pop"] == prf["unique_gen"] == prf["matching_types"] == 5


def test_disjoint_combinations_give_zero_prf():
    pop = pd.DataFrame({"a": [1, 1], "b": [1, 1]})
    gen = pd.DataFrame({"a": [2, 2], "b": [2, 2]})
    prf = precision_recall_f1(pop, gen)
    assert (prf["precision"], prf["recall"], prf["f1"]) == (0.0, 0.0, 0.0)


def test_partial_match_counts_rows_not_types():
    pop = pd.DataFrame({"a": [1, 1, 1, 2]})
    gen = pd.DataFrame({"a": [1, 3]})
    prf = precision_recall_f1(pop, gen)
    assert prf["precision"] == 0.5  # one of two synthetic rows exists in the target
    assert prf["recall"] == 0.75  # three of four target rows are reproduced
    assert prf["f1"] == round(2 * 0.5 * 0.75 / 1.25, 4)


def test_marginal_srmse_by_hand():
    pop = pd.DataFrame({"a": [1, 1, 2, 2]})  # proportions (0.5, 0.5)
    gen = pd.DataFrame({"a": [1, 1, 1, 2]})  # proportions (0.75, 0.25)
    expected = math.sqrt(((0.25) ** 2 + (0.25) ** 2) / 2) / 0.5
    s = srmse(pop, gen)
    assert s["marginal"] == pytest.approx(expected)
    assert math.isnan(s["bivariate"])  # one column, no pairs


def test_extra_empty_bins_scale_marginal_srmse():
    pop = pd.DataFrame({"a": [1, 1, 2, 2]})
    gen = pd.DataFrame({"a": [1, 1, 1, 2]})
    base = srmse(pop, gen)["marginal"]
    anchored = srmse(pop, gen, category_sets={"a": [1, 2, 3, 4]})["marginal"]
    assert anchored == pytest.approx(base * math.sqrt(4 / 2))


def test_member_table_repeats_household_attributes():
    hh = np.array([[1, 2], [3, 1]])
    ind = np.array([[5], [6], [7]])
    gid = np.array([1, 1, 2])
    df = member_table(hh, ind, gid)
    assert list(df.columns) == ["hh_1", "hh_2", "ind_1"]
    assert df.values.tolist() == [[1, 2, 5], [1, 2, 6], [3, 1, 7]]
    with pytest.raises(ValueError):
        member_table(hh, ind, np.array([0, 1, 2]))


def test_evaluate_synthetic_anchors_bins_to_target():
    target = _table()
    synthetic = pd.DataFrame({"a": [1, 2, 2], "b": [1, 1, 2]})
    out = evaluate_synthetic(target, synthetic, verbose=False)
    assert out["columns"] == ["a", "b"]
    assert out["cat_counts"] == {"a": 3, "b": 2}
    assert out["srmse"] == srmse(target, synthetic, category_sets=category_sets_from(target))
    assert out["prf"]["precision"] == 1.0
