from __future__ import annotations

import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def outrank_task_result_summary(args):
    triplets = pd.read_csv(
        os.path.join(args.output_folder, 'pairwise_ranks.tsv'), sep='\t',
    )
    triplets = triplets.sort_values(by=['Score'], ascending=False)
    final_ranking = []
    for enx, row in triplets.iterrows():
        final_row = None
        if args.label_column == row['FeatureA'].split('-')[0]:
            final_row = [row['FeatureB'], row['Score']]
        if args.label_column == row['FeatureB'].split('-')[0]:
            final_row = [row['FeatureA'], row['Score']]
        if final_row and args.label_column != final_row[0]:
            final_ranking.append(final_row)

    final_df = pd.DataFrame(final_ranking)
    final_df.columns = ['Feature', f'Score {args.heuristic}']
    final_df.index = np.arange(1, final_df.shape[0] + 1, 1)
    final_df = (
        final_df.groupby(by=['Feature'])
        .median()
        .reset_index()
        .sort_values(by=[f'Score {args.heuristic}'], ascending=False)
    )

    min_score = np.min(final_df[f'Score {args.heuristic}'].values)
    max_score = np.max(final_df[f'Score {args.heuristic}'].values)
    final_df[f'Score {args.heuristic}'] = (
        final_df[f'Score {args.heuristic}'] - min_score
    ) / (max_score - min_score)
    logging.info(f'Storing summary files to {args.output_folder}')
    pd.set_option('display.max_rows', None, 'display.max_columns', None)
    singles_path = os.path.join(args.output_folder, 'feature_singles.tsv')
    final_df = final_df.reset_index(drop=True)
    final_df.to_csv(singles_path, sep='\t')

    if args.interaction_order > 1:
        feature_store = defaultdict(list)
        for enx, row in final_df.iterrows():
            fname = row['Feature']
            score = row[f'Score {args.heuristic}']
            if 'AND' in fname:
                for el in fname.split('-')[0].split(' AND '):
                    feature_store[el].append(score)

        final_aggregate_df = []
        for k, v in feature_store.items():
            final_aggregate_df.append(
                {
                    'Feature': k,
                    f'Combined score (order: {args.interaction_order}, {args.heuristic})': np.median(
                        v,
                    ),
                },
            )
        final_aggregate_df = pd.DataFrame(final_aggregate_df)
        final_aggregate_df.to_csv(
            os.path.join(args.output_folder, 'feature_singles_aggregated.tsv'), sep='\t',
        )

    final_df = final_df[final_df['Feature'].str.contains('_tr_')]
    final_df.to_csv(
        singles_path.replace('.tsv', '_transformers_only_imp.tsv'), sep='\t',
    )
