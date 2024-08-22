# helper set of methods that enable anywhere verification of core functions
from __future__ import annotations

import logging
import os
import shutil
import subprocess

import pandas as pd

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('syn-logger')
logger.setLevel(logging.DEBUG)


def conduct_self_test(heuristic='MI-numba-randomized'):
    # Simulate full flow, ranking only
    subprocess.run(
        'outrank --task data_generator --num_synthetic_rows 100000', shell=True,
    )
    subprocess.run(
        f'outrank --task ranking --data_path test_data_synthetic --data_source csv-raw --heuristic {heuristic};',
        shell=True,
    )

    dfx = pd.read_csv('ranking_outputs/pairwise_ranks.tsv', sep='\t')

    logger.info("Verifying output's properties ..")
    assert dfx.shape[0] == 201
    assert dfx.shape[1] == 3
    assert dfx['FeatureA'].values.tolist().pop() == 'label-(2; 100)' or dfx['FeatureB'].values.tolist().pop() == 'label-(2; 100)'

    to_remove = ['ranking_outputs', 'test_data_synthetic']
    for path in to_remove:
        if os.path.exists(path) and os.path.isdir(path):
            logger.info(f'Removing {path} as part of cleanup ..')
            shutil.rmtree(path)

    logger.info(f'All tests passed for heuristic: {heuristic} \N{rocket}')


if __name__ == '__main__':
    conduct_self_test('MI-numba-randomized')
    conduct_self_test('max-value-coverage')
    logger.info('OutRank seems in shape \N{winking face}')
