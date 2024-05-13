from __future__ import annotations

import sys
import unittest

import jax
import optax
from flax import linen as nn
from jax import numpy as jnp
from jax import random

from outrank.algorithms.neural.mlp_nn import NNClassifier
sys.path.append('./outrank')


class NNClassifierTest(unittest.TestCase):

    def setUp(self):
        # Common setup operations, run before each test method
        self.learning_rate = 0.001
        self.architecture = [48, 48]
        self.epochs = 100
        self.clf = NNClassifier(
            self.learning_rate, self.architecture,
            self.epochs,
        )
        self.key = random.PRNGKey(7235123)

    def test_imports(self):
        # Test imports to ensure they are available
        try:
            import jax
            import optax
            import flax
        except ImportError as e:
            self.fail(f'Import failed: {e}')

    def test_initialization(self):
        # Check if the NNClassifier is initialized correctly
        self.assertIsInstance(self.clf, NNClassifier)
        self.assertEqual(self.clf.learning_rate, self.learning_rate)
        self.assertEqual(self.clf.architecture, self.architecture)
        self.assertEqual(self.clf.num_epochs, self.epochs)

    def test_self(self):
        # Check if `selftest` runs without assertion errors.
        try:
            self.clf.selftest()
        except AssertionError as e:
            self.fail(f'selftest() failed with an AssertionError: {e}')
