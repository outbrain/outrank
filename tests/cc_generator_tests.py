from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr

from outrank.algorithms.synthetic_data_generators.cc_generator import CategoricalClassification

@pytest.fixture
def cc_instance():
    return CategoricalClassification()

def test_init(cc_instance):
    assert cc_instance.dataset_info == ''

def test_generate_data_shape_and_type(cc_instance):
    X = cc_instance.generate_data(n_features=5, n_samples=100)
    assert isinstance(X, np.ndarray), 'Output should be a numpy array'
    assert X.shape == (100, 5), 'Shape should be (n_samples, n_features)'

def test_generate_data_cardinality(cc_instance):
    n_features = 5
    cardinality = 3
    X = cc_instance.generate_data(n_features=n_features, n_samples=100, cardinality=cardinality)
    unique_values = np.unique(X)
    assert len(unique_values) <= cardinality, 'Cardinality not respected for all features'

def test_generate_data_ensure_rep(cc_instance):
    n_features = 5
    cardinality = 50
    X = cc_instance.generate_data(n_features=n_features, n_samples=100, cardinality=cardinality, ensure_rep=True)
    unique_values = np.unique(X)
    assert len(unique_values) == cardinality, "Not all values represented when 'ensure_rep=True'"

def test_generate_feature_shape_and_type(cc_instance):
    feature = cc_instance._generate_feature(100, cardinality=5)
    assert isinstance(feature, np.ndarray), 'Output should be a numpy array'
    assert feature.shape == (100,), 'Shape should be (size,)'

def test_generate_feature_cardinality(cc_instance):
    feature = cc_instance._generate_feature(100, cardinality=5)
    unique_values = np.unique(feature)
    assert len(unique_values) <= 5, 'Feature cardinality not respected for all features'

def test_generate_feature_ensure_rep(cc_instance):
    feature = cc_instance._generate_feature(100, cardinality=50, ensure_rep=True)
    unique_values = np.unique(feature)
    assert len(unique_values) == 50, "Not all values represented when using 'ensure_rep=True'"

def test_generate_feature_values(cc_instance):
    values = [5, 6, 7, 8, 9, 10]
    feature = cc_instance._generate_feature(100, vec=values)
    unique_values = np.unique(feature)
    assert any(f in feature for f in values), 'Feature values not in input list'
def test_generate_feature_values_ensure_rep(cc_instance):
    values = [5, 6, 7, 8, 9, 10]
    feature = cc_instance._generate_feature(100, vec=values, ensure_rep=True)
    unique_values = np.unique(feature)
    assert (values == unique_values).all(), "Feature values should match input list when 'ensure_rep=True'"

def test_generate_feature_density(cc_instance):
    values = [0, 1, 2]
    p = [0.2, 0.4, 0.4]
    feature = cc_instance._generate_feature(10000, vec=values, ensure_rep=True, p=p)
    values, counts = np.unique(feature, return_counts=True)
    generated_p = np.round(counts/10000, decimals=1)
    assert (generated_p == p).all(), "Feature values should have density roughly equal to 'p'"

def test_generate_combinations_shape_and_type(cc_instance):
    X = cc_instance.generate_data(n_features=5, n_samples=100)
    indices = [0,1]
    X = cc_instance.generate_combinations(X, indices, combination_type='linear')
    assert isinstance(X, np.ndarray), 'Output should be a numpy array'
    assert X.shape == (100, 6), 'Shape should be (n_samples, n_features + 1)'

def test_generate_correlated_shape_and_type(cc_instance):
    X = cc_instance.generate_data(n_features=5, n_samples=100)
    indices = 0
    X = cc_instance.generate_correlated(X, indices, r=0.8)
    assert isinstance(X, np.ndarray), 'Output should be a numpy array'
    assert X.shape == (100, 6), 'Shape should be (n_samples, n_features + 1)'

def test_generate_correlated_correlaton(cc_instance):
    X = cc_instance.generate_data(n_features=5, n_samples=100)
    indices = 0
    X = cc_instance.generate_correlated(X, indices, r=0.8)
    Xt = X.T
    corr, _ = pearsonr(Xt[0], Xt[5])
    assert np.round(corr, decimals=1) == 0.8, "Resultant correlation should be equal to the 'r' parameter"


def test_generate_duplicates_shape_and_type(cc_instance):
    X = cc_instance.generate_data(n_features=5, n_samples=100)
    indices = 0
    X = cc_instance.generate_duplicates(X, indices)
    assert isinstance(X, np.ndarray), 'Output should be a numpy array'
    assert X.shape == (100, 6), 'Shape should be (n_samples, n_features + 1)'

def test_generate_duplicates_duplication(cc_instance):
    X = cc_instance.generate_data(n_features=5, n_samples=100)
    indices = 0
    X = cc_instance.generate_duplicates(X, indices)
    Xt = X.T
    assert (Xt[0] == Xt[-1]).all()

def test_xor_operation(cc_instance):
    a = np.array([1, 0, 1])
    b = np.array([0, 1, 1])
    arr = [a, b]
    result = cc_instance._xor(arr)
    expected = np.array([1, 1, 0])
    assert np.array_equal(result, expected), 'XOR operation did not produce expected result'

def test_and_operation(cc_instance):
    a = np.array([1, 0, 1])
    b = np.array([0, 1, 1])
    arr = [a, b]
    result = cc_instance._and(arr)
    expected = np.array([0, 0, 1])
    assert np.array_equal(result, expected), 'AND operation did not produce expected result'

def test_or_operation(cc_instance):
    a = np.array([1, 0, 1])
    b = np.array([0, 1, 1])
    arr = [a, b]
    result = cc_instance._or(arr)
    expected = np.array([1, 1, 1])
    assert np.array_equal(result, expected), 'OR operation did not produce expected result'

def test_generate_labels_shape_and_type(cc_instance):
    X = cc_instance.generate_data(n_features=5, n_samples=100)
    labels = cc_instance.generate_labels(X)
    assert isinstance(labels, np.ndarray), 'Output should be a numpy array'
    assert labels.shape == (100,), 'Shape should be (n_samples,)'

def test_generate_labels_distribution(cc_instance):
    X = cc_instance.generate_data(n_features=5, n_samples=100)
    labels = cc_instance.generate_labels(X, n=3, p=[0.2, 0.3, 0.5])
    unique, counts = np.unique(labels, return_counts=True)
    distribution = counts / 100
    expected_distribution = np.array([0.2, 0.3, 0.5])
    assert np.allclose(distribution, expected_distribution, atol=0.1), 'Label distribution does not match expected distribution'

def test_generate_labels_class_relation_linear(cc_instance):
    X = cc_instance.generate_data(n_features=5, n_samples=100)
    labels = cc_instance.generate_labels(X, class_relation='linear')
    assert isinstance(labels, np.ndarray), 'Output should be a numpy array'
    assert labels.shape == (100,), 'Shape should be (n_samples,)'

def test_generate_labels_class_relation_nonlinear(cc_instance):
    X = cc_instance.generate_data(n_features=5, n_samples=100)
    labels = cc_instance.generate_labels(X, class_relation='nonlinear')
    assert isinstance(labels, np.ndarray), 'Output should be a numpy array'
    assert labels.shape == (100,), 'Shape should be (n_samples,)'

def test_generate_labels_class_relation_cluster(cc_instance):
    X = cc_instance.generate_data(n_features=5, n_samples=100)
    labels = cc_instance.generate_labels(X, class_relation='cluster', balance=True)
    assert isinstance(labels, np.ndarray), 'Output should be a numpy array'
    assert labels.shape == (100,), 'Shape should be (n_samples,)'
