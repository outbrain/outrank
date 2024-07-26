from __future__ import annotations

import gzip
import os
from collections import Counter
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import tqdm

from outrank.core_utils import generic_line_parser
from outrank.core_utils import get_dataset_info
from outrank.core_utils import get_num_of_instances

try:
    import matplotlib.pyplot as plt
except:
    pass


def shannon_ent(string: str) -> float:
    counts = Counter(string)
    frequencies = ((i / len(string)) for i in counts.values())
    return -np.sum(f * np.log2(f) for f in frequencies)


def compute_entropy_avg(line: list) -> float:
    joint_ent = 0
    for field in line:
        joint_ent += shannon_ent(field)
    return joint_ent


def score_line(line):
    nan_prop = line.count('') / len(line)
    out_struct = {}
    out_struct['empty_string_prop'] = nan_prop
    out_struct['empty_dict'] = line.count('{}') / len(line)
    out_struct['all_empty'] = (line.count('{}') + line.count('')) / len(line)
    out_struct['all_zero'] = line.count('0') / len(line)
    for j in [30, 60, 100, 200, 300]:
        out_struct[f'all_more_{j}_chars'] = len(
            [x for x in line if len(x) > j], ) / len(line)
    out_struct['row_entropy'] = compute_entropy_avg(line)
    return out_struct


def outrank_task_rank_instances(args: Any) -> None:

    data_encoding = 'utf-8'
    delimiter = '\t'
    dataset_info = get_dataset_info(args)
    local_pbar = tqdm.tqdm(
        total=get_num_of_instances(dataset_info.data_path) - 1,
        position=0,
        disable=args.disable_tqdm == 'True',
    )
    local_pbar.set_description('Starting ranking computation')

    _, file_extension = os.path.splitext(dataset_info.data_path)

    if file_extension == '.gz':
        file_stream = gzip.open(
            dataset_info.data_path,
            'rt',
            encoding=data_encoding,
        )

    else:
        file_stream = open(dataset_info.data_path, encoding=data_encoding)
    line_counter = 0
    out_scores_lab = defaultdict(list)

    for line in file_stream:
        line_counter += 1
        local_pbar.update(1)

        parsed_line = generic_line_parser(
            line,
            delimiter,
            args,
            dataset_info.fw_map,
            dataset_info.column_names,
        )

        if line_counter > 100_000:
            break
        out_scores_lab[line[0]].append(score_line(parsed_line))

    for label, out_scores in out_scores_lab.items():
        out_df = pd.DataFrame(out_scores)
        os.makedirs(args.output_folder, exist_ok=True)
        for col in out_df.columns:
            sorted_vals = out_df[col].sort_values()
            plt.figure(figsize=(5, 5), dpi=300)
            plt.title(col + f' label: {label}')
            plt.hist(
                x=sorted_vals * 100,
                color='black',
                density=True,
                bins=100,
            )
            if 'entropy' not in col:
                plt.xlabel('Proportion of namespaces (%)')
            else:
                plt.xlabel('Row entropy')
            plt.ylabel('Density')
            plt.tight_layout()
            fname = f'distPlot{col}_{label}.pdf'
            plt.savefig(os.path.join(args.output_folder, fname), dpi=300)
            plt.cla()
            plt.clf()
