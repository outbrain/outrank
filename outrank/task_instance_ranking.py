from __future__ import annotations

import gzip
import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
import tqdm

from outrank.core_utils import generic_line_parser, get_dataset_info, get_num_of_instances

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def shannon_entropy(string: str) -> float:
    counts = Counter(string)
    frequencies = (i / len(string) for i in counts.values())
    return -sum(f * np.log2(f) for f in frequencies)

def compute_average_entropy(line: list[str]) -> float:
    return sum(shannon_entropy(field) for field in line)

def score_line(line: list[str]) -> dict[str, float]:
    total_fields = len(line)
    nan_prop = line.count('') / total_fields
    empty_dict_prop = line.count('{}') / total_fields
    all_empty_prop = (line.count('{}') + line.count('')) / total_fields
    all_zero_prop = line.count('0') / total_fields

    out_struct = {
        'empty_string_prop': nan_prop,
        'empty_dict': empty_dict_prop,
        'all_empty': all_empty_prop,
        'all_zero': all_zero_prop,
        'row_entropy': compute_average_entropy(line)
    }

    for j in [30, 60, 100, 200, 300]:
        out_struct[f'all_more_{j}_chars'] = sum(len(x) > j for x in line) / total_fields

    return out_struct

def outrank_task_rank_instances(args: Any) -> None:
    dataset_info = get_dataset_info(args)
    data_path = dataset_info.data_path
    data_encoding = 'utf-8'
    delimiter = '\t'

    total_lines = get_num_of_instances(data_path) - 1
    local_pbar = tqdm.tqdm(total=total_lines, position=0, disable=args.disable_tqdm == 'True')
    local_pbar.set_description('Starting ranking computation')

    _, file_extension = os.path.splitext(data_path)
    file_stream = gzip.open(data_path, 'rt', encoding=data_encoding) if file_extension == '.gz' else open(data_path, encoding=data_encoding)

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

    file_stream.close()

    os.makedirs(args.output_folder, exist_ok=True)
    for label, out_scores in out_scores_lab.items():
        out_df = pd.DataFrame(out_scores)
        for col in out_df.columns:
            sorted_vals = out_df[col].sort_values()
            plt.figure(figsize=(5, 5), dpi=300)
            plt.title(f'{col} label: {label}')
            plt.hist(
                x=sorted_vals * 100,
                color='black',
                density=True,
                bins=100,
            )
            plt.xlabel('Proportion of namespaces (%)' if 'entropy' not in col else 'Row entropy')
            plt.ylabel('Density')
            plt.tight_layout()
            fname = f'distPlot{col}_{label}.pdf'
            plt.savefig(os.path.join(args.output_folder, fname), dpi=300)
            plt.cla()
            plt.clf()

    local_pbar.close()
