from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hm_popsyn.io import prepare_from_merged_table
from hm_popsyn.pipeline import fit_and_generate


class PipelineTests(unittest.TestCase):
    def test_prepare_and_fit_generate_end_to_end(self) -> None:
        # Columns follow MATLAB merged table convention used in run scripts:
        # 0 hh_id, 1 ind_id, 2..5 household attrs, 6,7,8,10,11 individual attrs.
        # Small merged table with 4 households and 7 individuals, 4 household attributes
        # and 5 individual attributes.
        merged = np.array(
            [
                [101, 1, 1, 1, 1, 2, 1, 1, 1, 0, 1, 1],
                [101, 2, 1, 1, 1, 2, 2, 1, 2, 0, 2, 1],
                [102, 1, 1, 2, 2, 1, 1, 2, 1, 0, 2, 2],
                [102, 2, 1, 2, 2, 1, 2, 2, 2, 0, 1, 2],
                [103, 1, 2, 1, 2, 1, 1, 1, 2, 0, 2, 1],
                [104, 1, 2, 2, 1, 2, 2, 2, 1, 0, 1, 2],
                [104, 2, 2, 2, 1, 2, 1, 2, 2, 0, 2, 2],
            ],
            dtype=int,
        )

        prepared = prepare_from_merged_table(merged)
        self.assertEqual(prepared.grpdata.shape[1], 4)
        self.assertEqual(prepared.inddata.shape[1], 5)
        self.assertEqual(prepared.group_ids.shape[0], prepared.grpdata.shape[0])

        result = fit_and_generate(
            grpdata=prepared.grpdata,
            indgid=prepared.indgid,
            inddata=prepared.inddata,
            G=2,
            M=2,
            n_households=10,
            n_restarts=2,
            max_iter=120,
            tol=1e-7,
            seed=99,
            apply_rejection=True,
            age_col=0,
            employ_col=1,
        )

        self.assertEqual(result.synthetic_raw.household_data.shape[0], 10)
        self.assertEqual(result.synthetic_raw.household_data.shape[1], 4)
        self.assertEqual(result.synthetic_final.individual_data.shape[1], 5)
        self.assertTrue(np.all(result.synthetic_final.individual_group_id >= 1))
        self.assertTrue(np.allclose(np.sum(result.em_result.pi_g), 1.0, atol=1e-6))

    def test_prepare_and_fit_generate_larger_case_g5_m8(self) -> None:
        rng = np.random.default_rng(2026)

        # Construct 30 households and 83 individuals.
        n_households_obs = 30
        base = np.ones(n_households_obs, dtype=int)
        additional = rng.multinomial(
            83 - n_households_obs, np.full(n_households_obs, 1.0 / n_households_obs)
        )
        household_sizes = base + additional

        hh_ids = np.arange(1001, 1001 + n_households_obs, dtype=int)
        h1 = rng.integers(1, 5, size=n_households_obs)
        h2 = rng.integers(1, 4, size=n_households_obs)
        h3_size = household_sizes

        rows: list[list[int]] = []
        for i, hh in enumerate(hh_ids):
            for ind_id in range(1, int(household_sizes[i]) + 1):
                # 6 individual attributes, all 1-based categorical values.
                i_attrs = [
                    int(rng.integers(1, 7)),
                    int(rng.integers(1, 3)),
                    int(rng.integers(1, 6)),
                    int(rng.integers(1, 5)),
                    int(rng.integers(1, 4)),
                    int(rng.integers(1, 8)),
                ]
                rows.append([int(hh), ind_id, int(h1[i]), int(h2[i]), int(h3_size[i]), *i_attrs])

        merged = np.array(rows, dtype=int)
        self.assertEqual(merged.shape[0], 83)

        prepared = prepare_from_merged_table(
            merged,
            household_id_col=0,
            individual_id_col=1,
            household_attr_cols=(2, 3, 4),
            individual_attr_cols=(5, 6, 7, 8, 9, 10),
        )

        self.assertEqual(prepared.grpdata.shape, (30, 3))
        self.assertEqual(prepared.inddata.shape, (83, 6))
        self.assertEqual(prepared.group_ids.shape[0], 30)

        result = fit_and_generate(
            grpdata=prepared.grpdata,
            indgid=prepared.indgid,
            inddata=prepared.inddata,
            G=5,
            M=8,
            n_households=100,
            n_restarts=3,
            max_iter=180,
            tol=1e-7,
            seed=2026,
            apply_rejection=True,
            age_col=0,
            employ_col=1,
            household_size_col=2,
        )

        self.assertEqual(result.synthetic_raw.household_data.shape, (100, 3))
        self.assertEqual(result.synthetic_final.individual_data.shape[1], 6)
        self.assertTrue(np.all(result.synthetic_final.individual_group_id >= 1))
        final = result.synthetic_final
        self.assertTrue(np.all(final.individual_group_id <= final.household_data.shape[0]))

        self.assertEqual(result.em_result.pi_g.shape, (5,))
        self.assertEqual(result.em_result.pi_m.shape, (5, 8))
        self.assertTrue(np.allclose(np.sum(result.em_result.pi_g), 1.0, atol=1e-6))
        self.assertTrue(np.allclose(np.sum(result.em_result.pi_m, axis=1), 1.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
