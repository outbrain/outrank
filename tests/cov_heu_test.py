from __future__ import annotations

import sys
import unittest

import numpy as np

from outrank.algorithms.feature_ranking.ranking_cov_alignment import \
    max_pair_coverage

np.random.seed(123)
sys.path.append('./outrank')


class TestMaxPairCoverage(unittest.TestCase):
    def test_basic_functionality(self):
        array1 = np.array([1, 2, 3, 1, 2])
        array2 = np.array([4, 5, 6, 4, 5])
        result = max_pair_coverage(array1, array2)
        self.assertAlmostEqual(result, 2/5, places=5)

    def test_identical_elements(self):
        array1 = np.array([1, 1, 1, 1])
        array2 = np.array([1, 1, 1, 1])
        result = max_pair_coverage(array1, array2)
        self.assertEqual(result, 1.0)

    def test_large_arrays(self):
        array1 = np.random.randint(0, 100, size=10000)
        array2 = np.random.randint(0, 100, size=10000)
        result = max_pair_coverage(array1, array2)
        self.assertTrue(0 <= result <= 1)

    def test_all_unique_pairs(self):
        array1 = np.array([1, 2, 3, 4, 5])
        array2 = np.array([6, 7, 8, 9, 10])
        result = max_pair_coverage(array1, array2)
        self.assertEqual(result, 1/5)

    def test_all_same_pairs(self):
        array1 = np.array([1, 1, 1, 1, 1])
        array2 = np.array([2, 2, 2, 2, 2])
        result = max_pair_coverage(array1, array2)
        self.assertEqual(result, 1.0)

    def test_high_collision_potential(self):
        array1 = np.array([1] * 1000)
        array2 = np.array([2] * 1000)
        result = max_pair_coverage(array1, array2)
        self.assertEqual(result, 1.0)

    def test_very_large_arrays(self):
        array1 = np.random.randint(0, 1000, size=1000000)
        array2 = np.random.randint(0, 1000, size=1000000)
        result = max_pair_coverage(array1, array2)
        self.assertTrue(0 <= result <= 1)
