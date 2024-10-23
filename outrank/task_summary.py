from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def read_and_sort_triplets(triplets_path: str) -> pd.DataFrame:
    """Read triplets from a file and sort by the 'Score' column."""
    triplets = pd.read_csv(triplets_path, sep='\t')
    return triplets.sort_values(by='Score', ascending=False)


def generate_final_ranking(triplets: pd.DataFrame, label_column: str) -> list[list[Any]]:
    """Generate final ranking based on the label column."""
    final_ranking = []
    for _, row in triplets.iterrows():
        feature_a, feature_b = row['FeatureA'], row['FeatureB']
        score = row['Score']
        if label_column == feature_a.split('-')[0]:
            final_ranking.append([feature_b, score])
        elif label_column == feature_b.split('-')[0]:
            final_ranking.append([feature_a, score])
    return final_ranking


def create_final_dataframe(final_ranking: list[list[Any]], heuristic: str) -> pd.DataFrame:
    """Create a final DataFrame and normalize if necessary."""
    final_df = pd.DataFrame(final_ranking, columns=['Feature', f'Score {heuristic}'])
    final_df = (
        final_df.groupby('Feature')
        .median()
        .reset_index()
        .sort_values(by=f'Score {heuristic}', ascending=False)
    )

    if 'MI' in heuristic:
        min_score = final_df[f'Score {heuristic}'].min()
        max_score = final_df[f'Score {heuristic}'].max()
        final_df[f'Score {heuristic}'] = (final_df[f'Score {heuristic}'] - min_score) / (max_score - min_score)

    return final_df


def store_summary_files(final_df: pd.DataFrame, output_folder: str, heuristic: str, tldr: bool) -> None:
    """Store the summary files and optionally print the head of the DataFrame."""
    logging.info(f'Storing summary files to {output_folder}')
    pd.set_option('display.max_rows', None, 'display.max_columns', None)

    singles_path = os.path.join(output_folder, 'feature_singles.tsv')
    final_df.to_csv(singles_path, sep='\t', index=False)

    if tldr:
        print(final_df.head(20))


def handle_interaction_order(final_df: pd.DataFrame, output_folder: str, heuristic: str, interaction_order: int) -> None:
    """Handle the interaction order if it is greater than 1."""
    if interaction_order > 1:
        feature_store = defaultdict(list)
        for _, row in final_df.iterrows():
            fname = row['Feature']
            score = row[f'Score {heuristic}']
            if 'AND' in fname:
                for el in fname.split('-')[0].split(' AND '):
                    feature_store[el].append(score)

        final_aggregate_df = pd.DataFrame([
            {
                'Feature': k,
                f'Combined score (order: {interaction_order}, {heuristic})': np.median(v),
            }
            for k, v in feature_store.items()
        ])
        final_aggregate_df.to_csv(
            os.path.join(output_folder, 'feature_singles_aggregated.tsv'), sep='\t', index=False,
        )


def filter_transformers_only(final_df: pd.DataFrame, output_folder: str) -> None:
    """Filter the DataFrame to include only transformer features and store the result."""
    transformers_only_path = os.path.join(output_folder, 'feature_singles_transformers_only_imp.tsv')
    final_df[final_df['Feature'].str.contains('_tr_')].to_csv(transformers_only_path, sep='\t', index=False)


def outrank_task_result_summary(args) -> None:
    """Main function to generate a summary of outrank task results."""
    triplets_path = os.path.join(args.output_folder, 'pairwise_ranks.tsv')
    triplets = read_and_sort_triplets(triplets_path)

    final_ranking = generate_final_ranking(triplets, args.label_column)
    final_df = create_final_dataframe(final_ranking, args.heuristic)

    store_summary_files(final_df, args.output_folder, args.heuristic, args.tldr)
    handle_interaction_order(final_df, args.output_folder, args.heuristic, args.interaction_order)
    filter_transformers_only(final_df, args.output_folder)
