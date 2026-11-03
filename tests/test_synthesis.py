from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hm_popsyn.synthesis_eval import (
    generate_synthetic_population,
)


class SynthesisTests(unittest.TestCase):
    def test_generate_population_shapes(self) -> None:
        rng = np.random.default_rng(7)

        # G=2 household classes, M=2 individual classes
        pi_g = np.array([0.5, 0.5])
        pi_m = np.array([[0.7, 0.3], [0.2, 0.8]])

        # 4 household attributes (last one is household size: 1 or 2)
        phi_g = [
            np.array([[0.6, 0.3], [0.4, 0.7]]),
            np.array([[0.5, 0.8], [0.5, 0.2]]),
            np.array([[0.9, 0.4], [0.1, 0.6]]),
            np.array([[0.2, 0.1], [0.8, 0.9]]),
        ]

        # 2 individual attributes, universal classes
        phi_m = [
            np.array([[0.4, 0.8], [0.6, 0.2]]),
            np.array([[0.7, 0.3], [0.3, 0.7]]),
        ]

        res = generate_synthetic_population(
            n_households=10,
            pi_g=pi_g,
            pi_m=pi_m,
            phi_g=phi_g,
            phi_m=phi_m,
            household_size_col=3,
            rng=rng,
        )

        self.assertEqual(res.household_data.shape[0], 10)
        self.assertEqual(res.household_data.shape[1], 4)
        self.assertEqual(res.individual_data.shape[1], 2)
        self.assertEqual(res.individual_group_id.shape[0], res.individual_data.shape[0])
        self.assertTrue(np.all(res.household_class >= 1))
        self.assertTrue(np.all(res.individual_class >= 1))

    def test_generate_population_shapes_larger_g5_m8(self) -> None:
        rng = np.random.default_rng(77)
        G = 5
        M = 8

        pi_g = rng.dirichlet(np.ones(G))
        pi_m = rng.dirichlet(np.ones(M), size=G)

        # 4 household attributes, final one used as household size category.
        phi_g = []
        for n_cat in [4, 3, 5, 4]:
            mat = rng.gamma(shape=1.0, scale=1.0, size=(n_cat, G))
            phi_g.append(mat / np.sum(mat, axis=0, keepdims=True))

        # 6 individual attributes, universal individual classes M=8.
        phi_m = []
        for n_cat in [4, 4, 3, 5, 2, 4]:
            mat = rng.gamma(shape=1.0, scale=1.0, size=(n_cat, M))
            phi_m.append(mat / np.sum(mat, axis=0, keepdims=True))

        res = generate_synthetic_population(
            n_households=25,
            pi_g=pi_g,
            pi_m=pi_m,
            phi_g=phi_g,
            phi_m=phi_m,
            household_size_col=3,
            rng=rng,
        )

        self.assertEqual(res.household_data.shape, (25, 4))
        self.assertEqual(res.individual_data.shape[1], 6)
        self.assertEqual(res.individual_group_id.shape[0], res.individual_data.shape[0])
        self.assertTrue(np.all(res.household_class >= 1))
        self.assertTrue(np.all(res.household_class <= G))
        self.assertTrue(np.all(res.individual_class >= 1))
        self.assertTrue(np.all(res.individual_class <= M))
if __name__ == "__main__":
    unittest.main()
