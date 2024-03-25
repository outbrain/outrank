from __future__ import annotations

import sys
import unittest

from outrank.algorithms.sketches.counting_ultiloglog import (
    HyperLogLogWCache as HyperLogLog,
)

sys.path.append('./outrank')


class CompareStrategiesTest(unittest.TestCase):
    def test_hll_update(self):
        GLOBAL_CARDINALITY_STORAGE = dict()
        GLOBAL_CARDINALITY_STORAGE[1] = HyperLogLog(0.01)
        GLOBAL_CARDINALITY_STORAGE[1].add(123)
        GLOBAL_CARDINALITY_STORAGE[1].add(123)
        self.assertEqual(len(GLOBAL_CARDINALITY_STORAGE[1]), 1)

        GLOBAL_CARDINALITY_STORAGE[1].add(1232)
        self.assertEqual(len(GLOBAL_CARDINALITY_STORAGE[1]), 2)

        for j in range(100):
            GLOBAL_CARDINALITY_STORAGE[1].add(1232 + j)

        self.assertEqual(len(GLOBAL_CARDINALITY_STORAGE[1]), 101)

    def test_stress_multi_feature(self):
        GLOBAL_CARDINALITY_STORAGE = dict()
        for j in range(10):
            GLOBAL_CARDINALITY_STORAGE[j] = HyperLogLog(100000)
        for j in range(1000):
            for k in range(len(GLOBAL_CARDINALITY_STORAGE)):
                GLOBAL_CARDINALITY_STORAGE[k].add(1232 + j)

        for j in range(10):
            self.assertEqual(len(GLOBAL_CARDINALITY_STORAGE[j]), 1000)

    def test_stress_high_card(self):
        GLOBAL_CARDINALITY_STORAGE = dict()
        for j in range(10):
            GLOBAL_CARDINALITY_STORAGE[j] = HyperLogLog(0.01)

        for j in range(10000):
            for k in range(len(GLOBAL_CARDINALITY_STORAGE)):
                GLOBAL_CARDINALITY_STORAGE[k].add(1232 + j)

        # 1% err is toleratable above certain card range
        for j in range(10):
            self.assertLess(
                abs(len(GLOBAL_CARDINALITY_STORAGE[j]) - 10000), 100,
            )


if __name__ == '__main__':
    unittest.main()
