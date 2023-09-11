from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass

import numpy as np

from outrank.core_utils import parse_csv_raw
from outrank.core_utils import parse_namespace

sys.path.append('./outrank')


np.random.seed(123)
test_files_path = 'tests/tests_files'


@dataclass
class args:
    label_column: str = 'label'
    heuristic: str = 'surrogate-LR'
    target_ranking_only: bool = True
    interaction_order: int = 3
    combination_number_upper_bound: int = 1024


class CoreIOTest(unittest.TestCase):
    def test_parser_vw_namespace(self):
        float_set, _ = parse_namespace(
            os.path.join(test_files_path, 'vw_namespace_map.csv'),
        )
        expected_output = {f'f{x}' for x in [1, 2, 3]}

        self.assertEqual(float_set, expected_output)

    def test_parse_raw_csv(self):
        dataset_info = parse_csv_raw(test_files_path)
        self.assertEqual(dataset_info.column_names, ['f1', 'f2', 'f3', 'f4'])
        self.assertEqual(dataset_info.col_delimiter, ',')
        self.assertEqual(dataset_info.column_types, {})


if __name__ == '__main__':
    unittest.main()
