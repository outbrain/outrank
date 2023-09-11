from __future__ import annotations

import logging
import os

import pandas as pd

from outrank.visualizations.ranking_visualization import visualize_all

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def outrank_task_visualize_results(args):
    logging.info(f'Beginning visualization based on: {args.output_folder}.')

    triplets = pd.read_csv(
        os.path.join(args.output_folder, 'pairwise_ranks.tsv'), sep='\t',
    )
    visualize_all(
        triplets,
        args.output_folder,
        args.label_column,
        args.reference_model_JSON,
        image_format=args.image_format,
        heuristic=args.heuristic,
    )
