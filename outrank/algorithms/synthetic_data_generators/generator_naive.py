# This simplest thing we can do for now.
from __future__ import annotations

import numpy as np

np.random.seed(123)


def generate_random_matrix(num_features=100, size=20000):
    # random int matrix (categorical)
    sample = np.random.randint(10, 100, size=(size, num_features))

    target = sample[:, 30]
    # Some noise

    target[target < 20] = 0
    return sample, target


if __name__ == '__main__':
    import argparse
    import logging
    import os
    import shutil

    import pandas as pd

    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
    )
    logger = logging.getLogger('syn-logger')
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(
        description='Fast feature screening for sparse data sets.',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument('--output_df_name', type=str, default=None)

    parser.add_argument('--verify_outputs', type=str, default=None)

    parser.add_argument('--num_features', type=int, default=300)

    parser.add_argument('--size', type=int, default=1000)

    args = parser.parse_args()

    if args.output_df_name is not None:
        sample, target = generate_random_matrix(args.num_features, args.size)
        dfx = pd.DataFrame(sample)
        dfx.columns = [f'f{x}' for x in range(dfx.shape[1])]
        dfx['label'] = target
        if os.path.exists(args.output_df_name) and os.path.isdir(args.output_df_name):
            shutil.rmtree(args.output_df_name)
        os.mkdir(args.output_df_name)
        dfx.to_csv(f'./{args.output_df_name}/data.csv', index=False)

        logging.info(f'Generated dataset {dfx.shape} in {args.output_df_name}')
    elif args.verify_outputs is not None:
        rankings = pd.read_csv(
            os.path.join(args.verify_outputs, 'feature_singles.tsv'), sep='\t',
        )
        if rankings.iloc[1]['Feature'] != 'f30-(81; 100)':
            raise Exception(
                f'Could not retrieve the appropriate feature needle in the haystack {rankings.iloc[1].Feature}, exiting',
            )
        else:
            logger.info(
                f'Identified the appropriate feature in the haystack ({rankings.iloc[1].Feature})',
            )
