from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import pearsonr

from outrank.algorithms.synthetic_data_generators.cc_generator import CategoricalClassification

class TestCategoricalClassification(unittest.TestCase):

    def setUp(self):
        self.cc_instance = CategoricalClassification()

    def test_init(self):
        self.assertIsInstance(self.cc_instance, CategoricalClassification)
        dict = {
            'general': {},
            'combinations': [],
            'correlations': [],
            'duplicates': [],
            'labels': {},
            'noise': [],
        }
        self.assertEqual(self.cc_instance.dataset_info, dict)

    def test_generate_data_shape_and_type(self):
        X = self.cc_instance.generate_data(n_features=5, n_samples=100)
        self.assertIsInstance(X, np.ndarray, 'Output should be a numpy array')
        self.assertEqual(X.shape, (100, 5), 'Shape should be (n_samples, n_features)')

    def test_generate_data_cardinality(self):
        n_features = 5
        cardinality = 3
        X = self.cc_instance.generate_data(n_features=n_features, n_samples=100, cardinality=cardinality)
        unique_values = np.unique(X)
        self.assertLessEqual(len(unique_values), cardinality, 'Cardinality not respected for all features')

    def test_generate_data_ensure_rep(self):
        n_features = 5
        cardinality = 50
        X = self.cc_instance.generate_data(n_features=n_features, n_samples=100, cardinality=cardinality, ensure_rep=True)
        unique_values = np.unique(X)
        self.assertEqual(len(unique_values), cardinality, "Not all values represented when 'ensure_rep=True'")

    def test_generate_feature_shape_and_type(self):
        feature = self.cc_instance._generate_feature(100, cardinality=5)
        self.assertIsInstance(feature, np.ndarray, 'Output should be a numpy array')
        self.assertEqual(feature.shape, (100,), 'Shape should be (size,)')

    def test_generate_feature_cardinality(self):
        feature = self.cc_instance._generate_feature(100, cardinality=5)
        unique_values = np.unique(feature)
        self.assertLessEqual(len(unique_values), 5, 'Feature cardinality not respected for all features')

    def test_generate_feature_ensure_rep(self):
        feature = self.cc_instance._generate_feature(100, cardinality=50, ensure_rep=True)
        unique_values = np.unique(feature)
        self.assertEqual(len(unique_values), 50, "Not all values represented when using 'ensure_rep=True'")

    def test_generate_feature_values(self):
        values = [5, 6, 7, 8, 9, 10]
        feature = self.cc_instance._generate_feature(100, vec=values)
        unique_values = np.unique(feature)
        self.assertTrue(any(f in feature for f in values), 'Feature values not in input list')

    def test_generate_feature_values_ensure_rep(self):
        values = [5, 6, 7, 8, 9, 10]
        feature = self.cc_instance._generate_feature(100, vec=values, ensure_rep=True)
        unique_values = np.unique(feature)
        self.assertTrue(np.array_equal(values, unique_values), "Feature values should match input list when 'ensure_rep=True'")

    def test_generate_feature_density(self):
        values = [0, 1, 2]
        p = [0.2, 0.4, 0.4]
        feature = self.cc_instance._generate_feature(10000, vec=values, ensure_rep=True, p=p)
        values, counts = np.unique(feature, return_counts=True)
        generated_p = np.round(counts/10000, decimals=1)
        self.assertTrue(np.array_equal(generated_p, p), "Feature values should have density roughly equal to 'p'")

    def test_generate_combinations_shape_and_type(self):
        X = self.cc_instance.generate_data(n_features=5, n_samples=100)
        indices = [0,1]
        X = self.cc_instance.generate_combinations(X, indices, combination_type='linear')
        self.assertIsInstance(X, np.ndarray, 'Output should be a numpy array')
        self.assertEqual(X.shape, (100, 6), 'Shape should be (n_samples, n_features + 1)')

    def test_generate_correlated_shape_and_type(self):
        X = self.cc_instance.generate_data(n_features=5, n_samples=100)
        indices = 0
        X = self.cc_instance.generate_correlated(X, indices, r=0.8)
        self.assertIsInstance(X, np.ndarray, 'Output should be a numpy array')
        self.assertEqual(X.shape, (100, 6), 'Shape should be (n_samples, n_features + 1)')

    def test_generate_correlated_correlaton(self):
        X = self.cc_instance.generate_data(n_features=5, n_samples=100)
        indices = 0
        X = self.cc_instance.generate_correlated(X, indices, r=0.8)
        Xt = X.T
        corr, _ = pearsonr(Xt[0], Xt[5])
        self.assertAlmostEqual(np.round(corr, decimals=1), 0.8, "Resultant correlation should be equal to the 'r' parameter")

    def test_generate_duplicates_shape_and_type(self):
        X = self.cc_instance.generate_data(n_features=5, n_samples=100)
        indices = 0
        X = self.cc_instance.generate_duplicates(X, indices)
        self.assertIsInstance(X, np.ndarray, 'Output should be a numpy array')
        self.assertEqual(X.shape, (100, 6), 'Shape should be (n_samples, n_features + 1)')

    def test_generate_duplicates_duplication(self):
        X = self.cc_instance.generate_data(n_features=5, n_samples=100)
        indices = 0
        X = self.cc_instance.generate_duplicates(X, indices)
        Xt = X.T
        self.assertTrue((Xt[0] == Xt[-1]).all())

    def test_xor_operation(self):
        a = np.array([1, 0, 1])
        b = np.array([0, 1, 1])
        arr = np.array([a, b])
        result = self.cc_instance._xor(arr)
        expected = np.array([0, 0])
        self.assertTrue(np.array_equal(result, expected), 'XOR operation did not produce expected result')

    def test_and_operation(self):
        a = np.array([1, 0, 1])
        b = np.array([0, 1, 1])
        arr = np.array([a, b])
        result = self.cc_instance._and(arr)
        expected = np.array([0, 0])
        self.assertTrue(np.array_equal(result, expected), 'AND operation did not produce expected result')

    def test_or_operation(self):
        a = np.array([1, 0, 1])
        b = np.array([0, 1, 1])
        arr = np.array([a, b])
        result = self.cc_instance._or(arr)
        expected = np.array([1, 1])
        self.assertTrue(np.array_equal(result, expected), 'OR operation did not produce expected result')

    def test_generate_labels_shape_and_type(self):
        X = self.cc_instance.generate_data(n_features=5, n_samples=100)
        labels = self.cc_instance.generate_labels(X)
        self.assertIsInstance(labels, np.ndarray, 'Output should be a numpy array')
        self.assertEqual(labels.shape, (100,), 'Shape should be (n_samples,)')

    def test_generate_labels_distribution(self):
        X = self.cc_instance.generate_data(n_features=5, n_samples=100)
        labels = self.cc_instance.generate_labels(X, n=3, p=[0.2, 0.3, 0.5])
        unique, counts = np.unique(labels, return_counts=True)
        distribution = counts / 100
        # distribution = [round(d, 1) for d in distribution]
        expected_distribution = np.array([0.2, 0.3, 0.5])
        self.assertTrue(np.allclose(distribution, expected_distribution, rtol=0.1, atol=0.1), 'Label distribution does not match expected distribution')

    def test_generate_labels_class_relation_linear(self):
        X = self.cc_instance.generate_data(n_features=5, n_samples=100)
        labels = self.cc_instance.generate_labels(X, class_relation='linear')
        self.assertIsInstance(labels, np.ndarray, 'Output should be a numpy array')
        self.assertEqual(labels.shape, (100,), 'Shape should be (n_samples,)')

    def test_generate_labels_class_relation_nonlinear(self):
        X = self.cc_instance.generate_data(n_features=5, n_samples=100)
        labels = self.cc_instance.generate_labels(X, class_relation='nonlinear')
        self.assertIsInstance(labels, np.ndarray, 'Output should be a numpy array')
        self.assertEqual(labels.shape, (100,), 'Shape should be (n_samples,)')

    def test_generate_labels_class_relation_cluster(self):
        X = self.cc_instance.generate_data(n_features=5, n_samples=100)
        labels = self.cc_instance.generate_labels(X, class_relation='cluster')
        self.assertIsInstance(labels, np.ndarray, 'Output should be a numpy array')
        self.assertEqual(labels.shape, (100,), 'Shape should be (n_samples,)')
