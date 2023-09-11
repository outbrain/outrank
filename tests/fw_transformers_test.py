from __future__ import annotations

import sys
import unittest

import numpy as np

from outrank.feature_transformations.feature_transformer_vault.fw_transformers import (
    FW_TRANSFORMERS,
)

sys.path.append('./outrank')


class FWTransformersTest(unittest.TestCase):
    def test_log_probs(self):
        X = np.asarray([0.68294952, 0.7, 0.91263375])
        some_transformer = FW_TRANSFORMERS.get('_tr_fw_prob_log_res_1_gt_0.01')
        assert X is not None
        assert some_transformer is not None
        output = eval(some_transformer)
        self.assertListEqual(list(output), [-0.0, -0.0, -0.0])

    def test_sqrt_int_gt_1(self):
        X = np.asarray([1.0, 2.0, 5.0])
        some_transformer = FW_TRANSFORMERS.get('_tr_fw_sqrt_res_1_gt_1')
        assert X is not None
        assert some_transformer is not None
        output = eval(some_transformer)
        self.assertListEqual(list(output), [0.0, 1.0, 2.0])

    def test_sqrt_probs(self):
        X = np.asarray([0.68294952, 0.72944264, 0.91263375])
        some_transformer = FW_TRANSFORMERS.get(
            '_tr_fw_prob_sqrt_res_1_gt_0.01',
        )
        assert some_transformer is not None
        assert X is not None
        output = eval(some_transformer)
        self.assertListEqual(list(output), [1.0, 1.0, 1.0])

    def test_overall_transf_count(self):
        self.assertEqual(len(FW_TRANSFORMERS), 138)
