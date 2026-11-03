from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hm_popsyn.em import em_multilevel_across_universal


class EMTests(unittest.TestCase):
    def test_em_runs_and_returns_valid_probabilities(self) -> None:
        grpdata = np.array(
            [
                [1, 1, 1, 2],
                [1, 2, 2, 1],
                [2, 1, 2, 2],
                [2, 2, 1, 1],
            ],
            dtype=int,
        )

        # 8 individuals assigned to 4 households
        indgid = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=int)
        inddata = np.array(
            [
                [1, 1, 1, 1, 1],
                [2, 1, 2, 2, 1],
                [1, 2, 1, 2, 2],
                [2, 2, 2, 1, 2],
                [1, 1, 2, 1, 2],
                [2, 1, 1, 2, 1],
                [1, 2, 2, 2, 1],
                [2, 2, 1, 1, 2],
            ],
            dtype=int,
        )

        res = em_multilevel_across_universal(
            grpdata=grpdata,
            indgid=indgid,
            inddata=inddata,
            G=2,
            M=2,
            max_iter=120,
            tol=1e-7,
            rng=np.random.default_rng(123),
        )

        self.assertEqual(res.pi_g.shape, (2,))
        self.assertEqual(res.pi_m.shape, (2, 2))
        self.assertTrue(np.allclose(np.sum(res.pi_g), 1.0, atol=1e-6))
        self.assertTrue(np.allclose(np.sum(res.pi_m, axis=1), 1.0, atol=1e-6))
        self.assertGreaterEqual(res.likelihood.size, 1)
        self.assertEqual(res.wj.shape, (4, 2))
        self.assertEqual(res.x_jik.shape, (8, 2, 2))

    def test_em_larger_configuration_g5_m8(self) -> None:
        rng = np.random.default_rng(321)

        # 6 households with 4 group attributes (4th attribute is household size).
        household_sizes = np.array([3, 4, 2, 5, 3, 3], dtype=int)
        grpdata = np.column_stack(
            [
                rng.integers(1, 4, size=6),
                rng.integers(1, 4, size=6),
                rng.integers(1, 3, size=6),
                household_sizes,
            ]
        ).astype(int)

        # 20 individuals split across 6 households.
        indgid = np.repeat(np.arange(1, 7, dtype=int), household_sizes)
        inddata = rng.integers(1, 5, size=(20, 6), dtype=int)

        res = em_multilevel_across_universal(
            grpdata=grpdata,
            indgid=indgid,
            inddata=inddata,
            G=5,
            M=8,
            max_iter=180,
            tol=1e-7,
            rng=np.random.default_rng(321),
        )

        self.assertEqual(res.pi_g.shape, (5,))
        self.assertEqual(res.pi_m.shape, (5, 8))
        self.assertEqual(res.wj.shape, (6, 5))
        self.assertEqual(res.x_jik.shape, (20, 5, 8))

        self.assertTrue(np.all(np.isfinite(res.likelihood)))
        self.assertTrue(np.allclose(np.sum(res.pi_g), 1.0, atol=1e-6))
        self.assertTrue(np.allclose(np.sum(res.pi_m, axis=1), 1.0, atol=1e-6))
        self.assertTrue(np.allclose(np.sum(res.wj, axis=1), 1.0, atol=1e-6))
        self.assertTrue(np.all(np.isfinite(res.x_jik)))
        self.assertTrue(np.all(res.x_jik >= 0.0))
        self.assertTrue(np.all(res.x_jik <= 1.0 + 1e-10))


if __name__ == "__main__":
    unittest.main()
