from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hm_popsyn.gibbs import gibbs_universal


class GibbsTests(unittest.TestCase):
    def test_gibbs_runs_and_returns_posterior_shapes(self) -> None:
        grpdata = np.array(
            [
                [1, 1, 1, 2],
                [1, 2, 2, 1],
                [2, 1, 2, 2],
                [2, 2, 1, 1],
            ],
            dtype=int,
        )
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

        res = gibbs_universal(
            grpdata=grpdata,
            indgid=indgid,
            inddata=inddata,
            G=2,
            M=2,
            n_iter=40,
            burn_in=10,
            thin=3,
            seed=5,
        )

        self.assertEqual(res.pi_g_mean.shape, (2,))
        self.assertEqual(res.pi_m_mean.shape, (2, 2))
        self.assertEqual(res.z_trace.ndim, 2)
        self.assertEqual(res.c_trace.ndim, 2)
        self.assertTrue(np.allclose(np.sum(res.pi_g_mean), 1.0, atol=1e-6))
        self.assertTrue(np.allclose(np.sum(res.pi_m_mean, axis=1), 1.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
