from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

from outrank.core_ranking import compute_combined_features
from outrank.core_ranking import get_combinations_from_columns
from outrank.core_ranking import mixed_rank_graph
from outrank.feature_transformations.feature_transformer_vault import (
    default_transformers,
)
from outrank.feature_transformations.ranking_transformers import (
    FeatureTransformerGeneric,
)

sys.path.append('./outrank')


np.random.seed(123)
test_files_path = 'tests/tests_files'


@dataclass
class args:
    label_column: str = 'label'
    heuristic: str = 'surrogate-LR'
    target_ranking_only: str = 'True'
    interaction_order: int = 3
    combination_number_upper_bound: int = 1024


class CompareStrategiesTest(unittest.TestCase):
    def test_mixed_rank_graph_MI(self):
        initial_matrix = np.random.randint(0, 2, (1000, 5))
        dfx = pd.DataFrame(initial_matrix)
        dfx.columns = ['c' + str(x) for x in range(4)] + ['label']
        dfx['label'] = dfx['label'].astype(int)
        GLOBAL_CPU_POOL = Pool(processes=1)
        local_pbar = tqdm.tqdm(total=100, position=0)
        for heuristic in ['MI']:
            args.heuristic = heuristic
            ranking_triplets = mixed_rank_graph(
                dfx, args, GLOBAL_CPU_POOL, local_pbar,
            )
            unique_nodes = len({x[0] for x in ranking_triplets.triplet_scores})
            self.assertEqual(unique_nodes, dfx.shape[1])
            triplet_df = pd.DataFrame(ranking_triplets.triplet_scores)
            triplet_df.columns = ['f1', 'f2', 'score']
            self.assertEqual(int(np.std(triplet_df.score)), 0)

        GLOBAL_CPU_POOL.close()
        GLOBAL_CPU_POOL.join()

    def test_feature_transformer_generic(self):
        random_array = np.random.rand(100, 5)
        dfx = pd.DataFrame(random_array)
        numeric_column_names = dfx.columns
        transformer = FeatureTransformerGeneric(numeric_column_names)
        features_before = dfx.shape[1]
        transformed_df = transformer.construct_new_features(dfx)
        features_after = transformed_df.shape[1]
        self.assertEqual(features_after - features_before, 45)

    def test_transformer_generation(self):
        # Generic transformations commonly used
        default_ob_transformations = default_transformers.DEFAULT_TRANSFORMERS
        self.assertEqual(len(default_ob_transformations), 10)

    def test_compute_combinations(self):
        # Some random data - order=3 by default
        random_matrix = [[1, 2, 3], [3, 2, 1], [1, 1, 1], [2, 3, 4]]
        random_df = pd.DataFrame(random_matrix)
        random_df.columns = ['F1', 'F2', 'F3']
        local_pbar = tqdm.tqdm(total=100, position=0)
        transformed_df = compute_combined_features(
            random_df, None, args, local_pbar,
        )
        self.assertEqual(transformed_df.shape[1], 4)

        args.interaction_order = 2
        random_matrix = [[1, 2, 3], [3, 2, 1], [1, 1, 1], [2, 3, 4]]
        random_df = pd.DataFrame(random_matrix)
        random_df.columns = ['F1', 'F2', 'F3']
        transformed_df = compute_combined_features(
            random_df, None, args, local_pbar,
        )
        self.assertEqual(transformed_df.shape[1], 6)

    def test_get_combinations_from_columns_target_ranking_only(self):
        all_columns = pd.Index(['a', 'b', 'label'])
        args.heuristic = 'MI-numba-randomized'
        args.target_ranking_only = 'True'
        combinations = get_combinations_from_columns(all_columns, args)

        self.assertSetEqual(
            set(combinations),
            {('a', 'label'), ('b', 'label'), ('label', 'label')},
        )

    def test_get_combinations_from_columns(self):
        all_columns = pd.Index(['a', 'b', 'label'])
        args.heuristic = 'MI-numba-randomized'
        args.target_ranking_only = 'False'
        combinations = get_combinations_from_columns(all_columns, args)

        self.assertSetEqual(
            set(combinations),
            {('a', 'a'), ('b', 'b'), ('label', 'label'), ('a', 'b'), ('a', 'label'), ('b', 'label')},
        )

    def test_get_combinations_from_columns_3mr(self):
        all_columns = pd.Index(['a', 'b', 'label'])
        args.heuristic = 'MI-numba-3mr'
        combinations = get_combinations_from_columns(all_columns, args)

        self.assertSetEqual(
            set(combinations),
            {('a', 'a'), ('b', 'b'), ('label', 'label'), ('a', 'b'), ('a', 'label'), ('b', 'label')},
        )


if __name__ == '__main__':
    unittest.main()
