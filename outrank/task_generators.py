# OutRank is also capable of generating data sets.
from __future__ import annotations

import logging
import os
import shutil

import pandas as pd

from outrank.algorithms.synthetic_data_generators import generator_naive

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('syn-logger')
logger.setLevel(logging.DEBUG)


def outrank_task_generate_data_set(args):
    """Core method for generating data sets"""

    if args.generator_type == 'naive':
        sample, target = generator_naive.generate_random_matrix(
            args.num_synthetic_features, args.num_synthetic_rows,
        )
    else:
        raise ValueError(f'Generator {args.generator_type} not implemented.')

    dfx = pd.DataFrame(sample)
    dfx.columns = [f'f{x}' for x in range(dfx.shape[1])]
    dfx['label'] = target
    if os.path.exists(args.output_synthetic_df_name) and os.path.isdir(
        args.output_synthetic_df_name,
    ):
        logger.info(
            f'Found existing: {args.output_synthetic_df_name}, removing first ..',
        )
        shutil.rmtree(args.output_synthetic_df_name)
    os.mkdir(args.output_synthetic_df_name)
    dfx.to_csv(f'./{args.output_synthetic_df_name}/data.csv', index=False)

    logger.info(
        f'Generated data set of shape {dfx.shape} in {args.output_synthetic_df_name}',
    )
