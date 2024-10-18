from __future__ import annotations

import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def outrank_task_result_summary(args):
    triplets_path = os.path.join(args.output_folder, 'pairwise_ranks.tsv')
    triplets = pd.read_csv(triplets_path, sep='\t')
    triplets = triplets.sort_values(by='Score', ascending=False)

    final_ranking = []
    for _, row in triplets.iterrows():
        feature_a, feature_b = row['FeatureA'], row['FeatureB']
        score = row['Score']
        if args.label_column == feature_a.split('-')[0]:
            final_ranking.append([feature_b, score])
        elif args.label_column == feature_b.split('-')[0]:
            final_ranking.append([feature_a, score])

    final_df = pd.DataFrame(final_ranking, columns=['Feature', f'Score {args.heuristic}'])
    final_df = (
        final_df.groupby('Feature')
        .median()
        .reset_index()
        .sort_values(by=f'Score {args.heuristic}', ascending=False)
    )

    if "MI" in args.heuristic:
        min_score = final_df[f'Score {args.heuristic}'].min()
        max_score = final_df[f'Score {args.heuristic}'].max()
        final_df[f'Score {args.heuristic}'] = (final_df[f'Score {args.heuristic}'] - min_score) / (max_score - min_score)

    logging.info(f'Storing summary files to {args.output_folder}')
    pd.set_option('display.max_rows', None, 'display.max_columns', None)

    singles_path = os.path.join(args.output_folder, 'feature_singles.tsv')
    final_df.to_csv(singles_path, sep='\t', index=False)

    if args.interaction_order > 1:
        feature_store = defaultdict(list)
        for _, row in final_df.iterrows():
            fname = row['Feature']
            score = row[f'Score {args.heuristic}']
            if 'AND' in fname:
                for el in fname.split('-')[0].split(' AND '):
                    feature_store[el].append(score)

        final_aggregate_df = pd.DataFrame([
            {
                'Feature': k,
                f'Combined score (order: {args.interaction_order}, {args.heuristic})': np.median(v),
            }
            for k, v in feature_store.items()
        ])
        final_aggregate_df.to_csv(
            os.path.join(args.output_folder, 'feature_singles_aggregated.tsv'), sep='\t', index=False
        )

    transformers_only_path = singles_path.replace('.tsv', '_transformers_only_imp.tsv')
    final_df[final_df['Feature'].str.contains('_tr_')].to_csv(transformers_only_path, sep='\t', index=False)
