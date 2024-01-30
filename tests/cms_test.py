from __future__ import annotations

import sys
import unittest

import numpy as np

from outrank.algorithms.sketches.counting_cms import cms_hash
from outrank.algorithms.sketches.counting_cms import CountMinSketch


class TestCountMinSketch(unittest.TestCase):

    def setUp(self):
        # Set up a CountMinSketch instance with known parameters for testing
        self.depth = 6
        self.width = 2**10  # smaller width for testing purposes
        self.cms = CountMinSketch(self.depth, self.width)

    def test_init(self):
        self.assertEqual(self.cms.depth, self.depth)
        self.assertEqual(self.cms.width, self.width)
        self.assertEqual(self.cms.M.shape, (self.depth, self.width))
        self.assertEqual(len(self.cms.hash_seeds), self.depth)
        self.assertIsInstance(self.cms.tmp_vals, set)

    def test_add_and_query_single_element(self):
        # Test adding a single element and querying it
        element = 'test_element'
        self.cms.add(element)
        # The queried count should be at least 1 (could be higher due to hash collisions)
        self.assertGreaterEqual(self.cms.query(element), 1)

    def test_add_and_query_multiple_elements(self):
        elements = ['foo', 'bar', 'baz', 'qux', 'quux']
        for elem in elements:
            self.cms.add(elem)

        for elem in elements:
            self.assertGreaterEqual(self.cms.query(elem), 1)

    def test_batch_add_and_query(self):
        elements = ['foo', 'bar', 'baz'] * 10
        self.cms.batch_add(elements)

        for elem in set(elements):
            self.assertGreaterEqual(self.cms.query(elem), 10)

    def test_overflow_protection(self):
        # This test ensures that the set doesn't grow beyond its allowed size and memory usage
        for i in range(100001):
            self.cms.add(f'element{i}')

        self.assertLessEqual(len(self.cms.tmp_vals), 100000)
        self.assertLessEqual(sys.getsizeof(self.cms.tmp_vals) / (10 ** 3), 4200.0)

    def test_hash_uniformity(self):
        # Basic check for hash function's distribution
        seeds = np.array(np.random.randint(low=0, high=2**31 - 1, size=self.depth), dtype=np.uint32)
        hashes = [cms_hash(i, seeds[0], self.width) for i in range(1000)]
        # Expect fewer collisions over a small sample with a large width
        unique_hashes = len(set(hashes))
        self.assertGreater(unique_hashes, 900)
