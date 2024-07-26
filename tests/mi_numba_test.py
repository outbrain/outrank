from __future__ import annotations

import sys
import unittest

import numpy as np

from outrank.algorithms.feature_ranking.ranking_mi_numba import \
    mutual_info_estimator_numba

np.random.seed(123)
sys.path.append('./outrank')


class CompareStrategiesTest(unittest.TestCase):
    def test_mi_numba(self):
        a = np.random.random(10**6).reshape(-1).astype(np.int32)
        b = np.random.random(10**6).reshape(-1).astype(np.int32)
        final_score = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        self.assertEqual(final_score, 0.0)

    def test_mi_numba_random(self):
        a = np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=np.int32)
        b = np.random.random(8).reshape(-1).astype(np.int32)

        final_score = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        self.assertLess(final_score, 0.0)

    def test_mi_numba_mirror(self):
        a = np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=np.int32)
        b = np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=np.int32)
        final_score = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        self.assertGreater(final_score, 0.60)

    def test_mi_numba_longer_inputs(self):
        b = np.array([1, 0, 0, 0, 1, 1, 1, 0] * 10**5, dtype=np.int32)
        final_score = mutual_info_estimator_numba(b, b, np.float32(1.0), False)
        self.assertGreater(final_score, 0.60)

    def test_mi_numba_permutation(self):
        a = np.array([1, 0, 0, 0, 1, 1, 1, 0] * 10**3, dtype=np.int32)
        b = np.array(np.random.permutation(a), dtype=np.int32)
        final_score = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        self.assertLess(final_score, 0.05)

    def test_mi_numba_interaction(self):
        # Let's create incrementally more noisy features and compare
        a = np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=np.int32)
        lowest = np.array(np.random.permutation(a), dtype=np.int32)
        medium = np.array([1, 1, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        high = np.array([1, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

        lowest_score = mutual_info_estimator_numba(
            a, lowest, np.float32(1.0), False,
        )
        medium_score = mutual_info_estimator_numba(
            a, medium, np.float32(1.0), False,
        )
        high_score = mutual_info_estimator_numba(
            a, high, np.float32(1.0), False,
        )

        scores = [lowest_score, medium_score, high_score]
        sorted_score_indices = np.argsort(scores)
        self.assertEqual(np.sum(np.array([0, 1, 2]) - sorted_score_indices), 0)

    def test_mi_numba_higher_order(self):
        # The famous xor test
        vector_first = np.round(np.random.random(1000)).astype(np.int32)
        vector_second = np.round(np.random.random(1000)).astype(np.int32)
        vector_third = np.logical_xor(
            vector_first, vector_second,
        ).astype(np.int32)

        score_independent_first = mutual_info_estimator_numba(
            vector_first, vector_third, np.float32(1.0), False,
        )

        score_independent_second = mutual_info_estimator_numba(
            vector_second, vector_third, np.float32(1.0), False,
        )

        # This must be very close to zero/negative
        self.assertLess(score_independent_first, 0.01)
        self.assertLess(score_independent_second, 0.01)

        # --interaction_order 2 simulation
        combined_feature = np.array(
            list(hash(x) for x in zip(vector_first, vector_second)),
        ).astype(np.int32)

        score_combined = mutual_info_estimator_numba(
            combined_feature, vector_third, np.float32(1.0), False,
        )

        # This must be in the range of identity
        self.assertGreater(score_combined, 0.60)
