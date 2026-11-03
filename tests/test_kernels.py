"""
Test cases for the kernels in hm_popsyn.kernels.
Validates the lowest-level numerical helpers first. 
This confirms category indexing, grouped summation, one-hot sampling, and MATLAB-style
dimension summation are correct.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hm_popsyn.kernels import int_hist, mnrnd_new, sum_eik, sumdims


class KernelTests(unittest.TestCase):
    def test_int_hist(self) -> None:
        # Test that int_hist correctly counts occurrences of integers in the input array.
        x = np.array([1, 2, 2, 4, 1], dtype=float)
        got = int_hist(x, 4)
        exp = np.array([2.0, 2.0, 0.0, 1.0])
        np.testing.assert_allclose(got, exp)

    def test_sum_eik(self) -> None:
        # Test that sum_eik correctly sums elements along specified dimensions.
        eik = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 1.0]])
        mask = np.array([1, 2, 1])
        got = sum_eik(eik, mask, 2)
        exp = np.array([[2.0, 3.0], [3.0, 4.0]])
        np.testing.assert_allclose(got, exp)

    def test_mnrnd_new_one_hot(self) -> None:
        # Test that mnrnd_new correctly samples from a multinomial distribution.
        rng = np.random.default_rng(123)
        probs = np.array([[0.1, 0.9], [0.7, 0.3], [0.5, 0.5]])
        z = mnrnd_new(probs, rng)
        self.assertEqual(z.shape, probs.shape)
        self.assertTrue(np.all(z.sum(axis=1) == 1))

    def test_sumdims(self) -> None:
        # Test that sumdims correctly sums elements along specified dimensions.
        arr = np.ones((2, 3, 4))
        got = sumdims(arr, [2, 3])
        self.assertEqual(got.shape, (2,))
        np.testing.assert_allclose(got, np.array([12.0, 12.0]))


if __name__ == "__main__":
    unittest.main() # call unittest's test runner to execute the tests in this file
