from __future__ import annotations

import gzip
import itertools
import logging
import os
import random
import time
from collections import Counter
from collections import defaultdict
from collections import deque
from timeit import default_timer as timer
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import tqdm

from outrank.algorithms.importance_estimator import get_importances_estimate_pairwise
from outrank.algorithms.sketches.counting_ultiloglog import (
    HyperLogLogWCache as HyperLogLog,
)
from outrank.core_utils import BatchRankingSummary
from outrank.core_utils import extract_features_from_reference_JSON
from outrank.core_utils import generic_line_parser
from outrank.core_utils import internal_hash
from outrank.core_utils import NominalFeatureSummary
from outrank.core_utils import NumericFeatureSummary
from outrank.feature_transformations.ranking_transformers import FeatureTransformerGeneric
from outrank.feature_transformations.ranking_transformers import FeatureTransformerNoise

logger = logging.getLogger('syn-logger')
logger.setLevel(logging.DEBUG)
random.seed(a=123, version=2)
GLOBAL_CARDINALITY_STORAGE: dict[Any, Any] = dict()
GLOBAL_RARE_VALUE_STORAGE: dict[str, Any] = Counter()

IGNORED_VALUES = set()
HYPERLL_ERROR_BOUND = 0.02


def encode_int_column(input_tuple: tuple[str, Any]) -> tuple[Any, list[int]]:
    """Encode column values as categoric (at a batch level!)"""

    hashes, _ = pd.factorize(input_tuple[1])
    return input_tuple[0], hashes


def mixed_rank_graph(
    input_dataframe: pd.DataFrame, args: Any, cpu_pool: Any, pbar: Any,
) -> BatchRankingSummary:
    """Compute the full mixed rank graph corresponding to all pairwise feature interactions based on the selected heuristic"""

    all_columns = input_dataframe.columns

    triplets = []
    tmp_df = input_dataframe.copy()
    out_time_struct = {}

    # Handle cont. types prior to interaction evaluation
    pbar.set_description('Encoding columns')
    jobs = [(cname, tmp_df[cname]) for cname in all_columns]
    col_dots = '.'
    start_enc_timer = timer()
    with cpu_pool as p:
        results = p.amap(encode_int_column, jobs)
        while not results.ready():
            time.sleep(4)
            col_dots = col_dots + '.'
            pbar.set_description(f'Encoding columns .{col_dots}')
        tmp_df = pd.DataFrame({k: v for k, v in results.get()})
    end_enc_timer = timer()
    out_time_struct['encoding_columns'] = end_enc_timer - start_enc_timer

    # Helper method for parallel estimation
    combinations = list(
        itertools.combinations_with_replacement(all_columns, 2),
    )

    if '3mr' in args.heuristic:
        rel_columns = [
            column for column in all_columns if ' AND_REL ' in column
        ]
        non_rel_columns = list(set(all_columns) - set(rel_columns))
        combinations = list(
            itertools.combinations_with_replacement(non_rel_columns, 2),
        )
        combinations += [(column, args.label_column) for column in rel_columns]
    else:
        combinations = list(
            itertools.combinations_with_replacement(all_columns, 2),
        )

    # Diagonal elements
    for individual_column in all_columns:
        if individual_column != args.label_column:
            combinations += [(individual_column, individual_column)]

    # Some applications do not require the full feature-feature triangular matrix
    if (args.target_ranking_only == 'True') and ('3mr' not in args.heuristic):
        combinations = [x for x in combinations if args.label_column in x]

    random.shuffle(combinations)
    combinations = combinations[: args.combination_number_upper_bound]

    if args.heuristic == 'Constant':
        final_constant_imp = []
        for c1, c2 in combinations:
            final_constant_imp.append((c1, c2, 0.0))

        out_time_struct['feature_score_computation'] = end_enc_timer - \
            start_enc_timer
        return BatchRankingSummary(final_constant_imp, out_time_struct)

    # Map the scoring calls to the worker pool
    pbar.set_description('Allocating thread pool')

    # starmap is an alternative that is slower unfortunately (but nicer)
    def get_grounded_importances_estimate(combination: tuple[str]) -> Any:
        return get_importances_estimate_pairwise(combination, args, tmp_df=tmp_df)

    start_enc_timer = timer()
    with cpu_pool as p:
        pbar.set_description(f'Computing (#ftr={len(combinations)})')
        results = p.amap(get_grounded_importances_estimate, combinations)
        while not results.ready():
            time.sleep(4)
        triplets = results.get()
    end_enc_timer = timer()
    out_time_struct['feature_score_computation'] = end_enc_timer - \
        start_enc_timer

    # Gather the final triplets
    pbar.set_description('Aggregation of ranking results')
    final_triplets = []
    for triplet in triplets:
        inv = (triplet[1], triplet[0], triplet[2])
        final_triplets.append(inv)
        final_triplets.append(triplet)
        triplets = final_triplets

    pbar.set_description('Proceeding to the next batch of data')
    return BatchRankingSummary(triplets, out_time_struct)


def enrich_with_transformations(
    input_dataframe: pd.DataFrame, num_col_types: set[str], logger: Any, args: Any,
) -> pd.DataFrame:
    """Construct a collection of new features based on pre-defined transformations/rules"""

    transformer = FeatureTransformerGeneric(
        num_col_types, preset=args.transformers,
    )
    transformed_df = transformer.construct_new_features(input_dataframe)
    logger.info(
        f'Constructed {len(transformer.constructed_feature_names)} new features ..',
    )

    return transformed_df


def compute_combined_features(
    input_dataframe: pd.DataFrame,
    logger: Any,
    args: Any,
    pbar: Any,
    is_3mr: bool = False,
) -> pd.DataFrame:
    """Compute higher order features via xxhash-based trick."""

    all_columns = [
        x for x in input_dataframe.columns if x != args.label_column
    ]
    join_string = ' AND_REL ' if is_3mr else ' AND '
    interaction_order = 2 if is_3mr else args.interaction_order

    full_combination_space = list(
        itertools.combinations(all_columns, interaction_order),
    )

    if args.combination_number_upper_bound:
        random.shuffle(full_combination_space)
        full_combination_space = full_combination_space[
            : args.combination_number_upper_bound
        ]

    com_counter = 0
    new_feature_hash = {}
    for new_combination in full_combination_space:
        pbar.set_description(
            f'Created {com_counter}/{len(full_combination_space)}',
        )
        combined_feature: list[str] = [str(0)] * input_dataframe.shape[0]
        for feature in new_combination:
            tmp_feature = input_dataframe[feature].tolist()
            for enx, el in enumerate(tmp_feature):
                combined_feature[enx] = str(
                    internal_hash(
                        str(combined_feature[enx]) + str(el),
                    ),
                )
        ftr_name = join_string.join(str(x) for x in new_combination)
        new_feature_hash[ftr_name] = combined_feature
        com_counter += 1
    tmp_df = pd.DataFrame(new_feature_hash)
    pbar.set_description('Concatenating into final frame ..')
    input_dataframe = pd.concat([input_dataframe, tmp_df], axis=1)
    del tmp_df

    return input_dataframe


def compute_expanded_multivalue_features(
    input_dataframe: pd.DataFrame, logger: Any, args: Any, pbar: Any,
) -> pd.DataFrame:
    """Compute one-hot encoded feature space based on each designated multivalue feature. E.g., feature with value "a,b,c" becomes three features, values of which are presence of a given value in a mutlivalue feature of choice."""

    considered_multivalue_features = args.explode_multivalue_features.split(
        ';',
    )
    new_feature_hash = {}
    missing_symbols = set(args.missing_value_symbols.split(','))

    for multivalue_feature in considered_multivalue_features:
        multivalue_feature_vector = input_dataframe[multivalue_feature].values.tolist(
        )
        multivalue_feature_vector = [
            x.replace(',', '-') for x in multivalue_feature_vector
        ]
        multivalue_sets = [
            set(x.split('-'))
            for x in multivalue_feature_vector
        ]
        unique_values = set.union(*multivalue_sets)

        for missing_symbol in missing_symbols:
            if missing_symbol in unique_values:
                unique_values.remove(missing_symbol)

        for unique_value in unique_values:
            tmp_vec = []
            for enx, multivalue in enumerate(multivalue_sets):
                if unique_value in multivalue:
                    tmp_vec.append('1')
                else:
                    tmp_vec.append('')

            new_feature_hash[f'MULTIEX-{multivalue_feature}-{unique_value}'] = tmp_vec

    tmp_df = pd.DataFrame(new_feature_hash)
    input_dataframe = pd.concat([input_dataframe, tmp_df], axis=1)
    del tmp_df

    return input_dataframe


def compute_subfeatures(
    input_dataframe: pd.DataFrame, logger: Any, args: Any, pbar: Any,
) -> pd.DataFrame:
    """Compute derived features that are more fine-grained. Implements logic around two operators that govern feature construction.
    ->: One sided construction - every value from left side is fine, separate ones from the right side feature will be considered.
    <->: Two sided construction - two-sided values present. This means that each value from a is combined with each from b, forming |A|*|B| new features (one-hot encoded)
    """

    all_subfeature_pair_seeds = args.subfeature_mapping.split(';')
    new_feature_hash = dict()

    for seed_pair in all_subfeature_pair_seeds:
        if '<->' in seed_pair:
            feature_first, feature_second = seed_pair.split('<->')

        elif '->' in seed_pair:
            feature_first, feature_second = seed_pair.split('->')

        else:
            raise NotImplementedError(
                'Please specify valid subfeature operator (<-> or ->)',
            )

        subframe = input_dataframe[[feature_first, feature_second]]
        unique_feature_second = subframe[feature_second].unique()
        feature_first_vec = subframe[feature_first].tolist()
        feature_second_vec = subframe[feature_second].tolist()
        out_template_feature = [
            (a, b) for a, b in zip(feature_first_vec, feature_second_vec)
        ]

        if '<->' in seed_pair:
            unique_feature_first = subframe[feature_first].unique()

            mask_types = []
            for unique_target_feature_value in unique_feature_second:
                for unique_seed_feature_value in unique_feature_first:
                    mask_types.append(
                        (unique_seed_feature_value, unique_target_feature_value),
                    )

            for mask_type in mask_types:
                new_feature = []
                for value_tuple in out_template_feature:
                    if (
                        value_tuple[0] == mask_type[0]
                        and value_tuple[1] == mask_type[1]
                    ):
                        new_feature.append(str(1))
                    else:
                        new_feature.append(str(0))
                feature_name = (
                    f'SUBFEATURE|{feature_first}|{feature_second}-'
                    + mask_type[0]
                    + '&'
                    + mask_type[1]
                )
                new_feature_hash[feature_name] = new_feature

            del new_feature

        elif '->' in seed_pair:
            for unique_target_feature_value in unique_feature_second:
                tmp_new_feature = [
                    'AND'.join(
                        x,
                    ) if x[1] == unique_target_feature_value else ''
                    for x in out_template_feature
                ]
                feature_name_final = (
                    'SUBFEATURE-' + feature_first + '&' + unique_target_feature_value
                )
                new_feature_hash[feature_name_final] = tmp_new_feature

    tmp_df = pd.DataFrame(new_feature_hash)
    input_dataframe = pd.concat([input_dataframe, tmp_df], axis=1)

    del tmp_df
    return input_dataframe


def include_noisy_features(
    input_dataframe: pd.DataFrame, logger: Any, args: Any,
) -> pd.DataFrame:
    """Add randomized features that serve as a sanity check"""

    transformer = FeatureTransformerNoise()
    transformed_df = transformer.construct_new_features(
        input_dataframe, args.label_column,
    )

    return transformed_df


def compute_coverage(input_dataframe: pd.DataFrame, args: Any) -> dict[str, set[str]]:
    """Compute coverage of features, incrementally"""
    output_storage_cov = defaultdict(set)
    all_missing_symbols = set(args.missing_value_symbols.split(','))
    for column in input_dataframe:
        all_missing = sum(
            [
                input_dataframe[column].values.tolist().count(x)
                for x in all_missing_symbols
            ],
        )

        output_storage_cov[column] = (
            1 - (all_missing / input_dataframe.shape[0])
        ) * 100

    return output_storage_cov


def compute_feature_memory_consumption(input_dataframe: pd.DataFrame, args: Any) -> dict[str, set[str]]:
    """An approximation of how much feature take up"""
    output_storage_features = defaultdict(set)
    for col in input_dataframe.columns:
        specific_column = [
            str(x).strip() for x in input_dataframe[col].astype(str).values.tolist()
        ]
        col_size = sum(
            len(x.encode())
            for x in specific_column
        ) / input_dataframe.shape[0]
        output_storage_features[col] = col_size
    return output_storage_features


def compute_value_counts(input_dataframe: pd.DataFrame, args: Any):
    """Update the count structure"""

    global GLOBAL_RARE_VALUE_STORAGE
    global IGNORED_VALUES

    for column in input_dataframe.columns:
        main_values = input_dataframe[column].values
        for value in main_values:
            if value not in IGNORED_VALUES:
                GLOBAL_RARE_VALUE_STORAGE.update({(column, value): 1})

    for key, val in GLOBAL_RARE_VALUE_STORAGE.items():
        if val > args.rare_value_count_upper_bound:
            IGNORED_VALUES.add(key)

    for to_remove_val in IGNORED_VALUES:
        del GLOBAL_RARE_VALUE_STORAGE[to_remove_val]


def compute_cardinalities(input_dataframe: pd.DataFrame, pbar: Any) -> None:
    """Compute cardinalities of features, incrementally"""

    global GLOBAL_CARDINALITY_STORAGE
    output_storage_card = defaultdict(set)
    for enx, column in enumerate(input_dataframe):
        output_storage_card[column] = set(input_dataframe[column].unique())
        if column not in GLOBAL_CARDINALITY_STORAGE:
            GLOBAL_CARDINALITY_STORAGE[column] = HyperLogLog(
                HYPERLL_ERROR_BOUND,
            )

        for unique_value in set(input_dataframe[column].unique()):
            if unique_value:
                GLOBAL_CARDINALITY_STORAGE[column].add(
                    internal_hash(unique_value),
                )
        pbar.set_description(
            f'Computing cardinality (Hyperloglog update) {enx}/{input_dataframe.shape[1]}',
        )


def compute_bounds_increment(
    input_dataframe: pd.DataFrame, numeric_column_types: set[str],
) -> dict[str, Any]:
    all_features = input_dataframe.columns
    numeric_column_types = set(numeric_column_types)
    summary_object = {}
    summary_storage: Any = {}
    for feature in all_features:
        if feature in numeric_column_types:
            feature_vector = pd.to_numeric(
                input_dataframe[feature], errors='coerce',
            )
            minimum = np.min(feature_vector)
            maximum = np.max(feature_vector)
            mean = np.mean(feature_vector)
            summary_storage = NumericFeatureSummary(
                feature, minimum, maximum, mean, len(
                    np.unique(feature_vector),
                ),
            )
            summary_object[feature] = summary_storage

        else:
            feature_vector = input_dataframe[feature].values
            summary_storage = NominalFeatureSummary(
                feature, len(np.unique(feature_vector)),
            )
            summary_object[feature] = summary_storage

    return summary_object


def compute_batch_ranking(
    line_tmp_storage: list[list[Any]],
    numeric_column_types: set[str],
    args: Any,
    cpu_pool: Any,
    column_descriptions: list[str],
    logger: Any,
    pbar: Any,
) -> tuple[BatchRankingSummary, dict[str, Any], dict[str, set[str]], dict[str, set[str]]]:
    """Enrich the feature space and compute the batch importances"""

    input_dataframe = pd.DataFrame(line_tmp_storage)
    input_dataframe.columns = column_descriptions
    pbar.set_description('Control features')

    if args.feature_set_focus:
        if args.feature_set_focus == '_all_from_reference_JSON':
            focus_set = extract_features_from_reference_JSON(
                args.reference_model_JSON,
            )

        else:
            focus_set = set(args.feature_set_focus.split(','))

        focus_set.add(args.label_column)
        focus_set = {x for x in focus_set if x in input_dataframe.columns}
        input_dataframe = input_dataframe[focus_set]

    if args.transformers != 'none':
        pbar.set_description('Adding transformations')
        input_dataframe = enrich_with_transformations(
            input_dataframe, numeric_column_types, logger, args,
        )

    if args.explode_multivalue_features != 'False':
        pbar.set_description('Constructing new features from multivalue ones')
        input_dataframe = compute_expanded_multivalue_features(
            input_dataframe, logger, args, pbar,
        )

    if args.subfeature_mapping != 'False':
        pbar.set_description('Constructing new (sub)features')
        input_dataframe = compute_subfeatures(
            input_dataframe, logger, args, pbar,
        )

    if args.interaction_order > 1:
        pbar.set_description('Constructing new features')
        input_dataframe = compute_combined_features(
            input_dataframe, logger, args, pbar,
        )

    # in case of 3mr we compute the score of combinations against the target
    if '3mr' in args.heuristic:
        pbar.set_description(
            'Constructing features for computing relations in 3mr',
        )
        input_dataframe = compute_combined_features(
            input_dataframe, logger, args, pbar, True,
        )

    if args.include_noise_baseline_features == 'True' and args.heuristic != 'Constant':
        pbar.set_description('Computing baseline features')
        input_dataframe = include_noisy_features(input_dataframe, logger, args)

    # Compute incremental statistic useful for data inspection/transformer generation
    pbar.set_description('Computing coverage')
    coverage_storage = compute_coverage(input_dataframe, args)
    feature_memory_consumption = compute_feature_memory_consumption(
        input_dataframe, args,
    )
    compute_cardinalities(input_dataframe, pbar)

    if args.task == 'identify_rare_values':
        compute_value_counts(input_dataframe, args)

    bounds_storage = compute_bounds_increment(
        input_dataframe, numeric_column_types,
    )

    pbar.set_description(
        f'Computing ranks for {input_dataframe.shape[1]} features',
    )

    return (
        mixed_rank_graph(input_dataframe, args, cpu_pool, pbar),
        bounds_storage,
        coverage_storage,
        feature_memory_consumption,
    )


def get_num_of_instances(fname: str) -> int:
    """Count the number of lines in a file, fast - useful for progress logging"""

    def _make_gen(reader):
        while True:
            b = reader(2**16)
            if not b:
                break
            yield b

    with open(fname, 'rb') as f:
        count = sum(buf.count(b'\n') for buf in _make_gen(f.raw.read))
    return count


def get_grouped_df(importances_df_list: list[tuple[str, str, float]]) -> pd.DataFrame:
    """A helper method that enables median-based aggregation after processing"""

    importances_df = pd.DataFrame(importances_df_list)
    if len(importances_df) == 0:
        return None
    importances_df.columns = ['FeatureA', 'FeatureB', 'Score']
    grouped = importances_df.groupby(
        ['FeatureA', 'FeatureB'],
    ).median().reset_index()
    return grouped


def checkpoint_importances_df(importances_batch: list[tuple[str, str, float]]) -> None:
    """A helper which stores intermediary state - useful for longer runs"""

    gdf = get_grouped_df(importances_batch)
    if gdf is not None:
        gdf.to_csv('ranking_checkpoint_tmp.tsv', sep='\t')


def estimate_importances_minibatches(
    input_file: str,
    column_descriptions: list,
    fw_col_mapping: dict[str, str],
    numeric_column_types: set,
    batch_size: int = 100000,
    args: Any = None,
    data_encoding: str = 'utf-8',
    cpu_pool: Any = None,
    delimiter: str = '\t',
    feature_construction_mode: bool = False,
    logger: Any = None,
) -> tuple[list[dict[str, Any]], Any, dict[Any, Any], list[dict[str, Any]], list[dict[str, set[str]]], defaultdict[str, list[set[str]]], dict[str, Any]]:
    """Interaction score estimator - suitable for example for csv-like input data types.
    This type of data is normally a single large csv, meaning that minibatch processing needs to
    happen during incremental handling of the file (that"s not the case for pre-separated ob data)
    """

    invalid_line_queue: Any = deque([], maxlen=2**5)

    invalid_lines = 0
    line_counter = 0

    importances_df: list[Any] = []
    line_tmp_storage = []
    bounds_storage_batch = []
    memory_storage_batch = []
    step_timing_checkpoints = []

    local_coverage_object = defaultdict(list)
    local_pbar = tqdm.tqdm(
        total=get_num_of_instances(input_file) - 1, position=0,
    )

    file_name, file_extension = os.path.splitext(input_file)

    if file_extension == '.gz':
        file_stream = gzip.open(input_file, 'rt', encoding=data_encoding)

    else:
        file_stream = open(input_file, encoding=data_encoding)

    file_stream.readline()

    local_pbar.set_description('Starting ranking computation')
    for line in file_stream:
        line_counter += 1
        local_pbar.update(1)

        if line_counter % args.subsampling != 0:
            continue

        parsed_line = generic_line_parser(
            line, delimiter, args, fw_col_mapping, column_descriptions,
        )

        if len(parsed_line) == len(column_descriptions):
            line_tmp_storage.append(parsed_line)

        else:
            invalid_line_queue.appendleft(str(parsed_line))
            invalid_lines += 1

        # Batches need to be processed on-the-fly
        if len(line_tmp_storage) >= args.minibatch_size:

            importances_batch, bounds_storage, coverage_storage, memory_storage = compute_batch_ranking(
                line_tmp_storage,
                numeric_column_types,
                args,
                cpu_pool,
                column_descriptions,
                logger,
                local_pbar,
            )

            bounds_storage_batch.append(bounds_storage)
            memory_storage_batch.append(memory_storage)
            for k, v in coverage_storage.items():
                local_coverage_object[k].append(v)

            del coverage_storage

            line_tmp_storage = []
            step_timing_checkpoints.append(importances_batch.step_times)
            importances_df += importances_batch.triplet_scores

            if args.heuristic != 'Constant':
                local_pbar.set_description('Creating checkpoint')
                checkpoint_importances_df(importances_df)

    file_stream.close()

    local_pbar.set_description('Parsing the remainder')
    if invalid_lines > 0:
        logger.info(
            f"Detected {invalid_lines} invalid lines. If this number is very high, it's possible your header is off - re-check your data/attribute-feature mappings please!",
        )

        invalid_lines_log = '\n INVALID_LINE ====> '.join(
            list(invalid_line_queue)[0:5],
        )
        logger.info(
            f'5 samples of invalid lines are printed below\n {invalid_lines_log}',
        )

    remaining_batch_size = len(line_tmp_storage)

    if remaining_batch_size > 2**10:
        line_tmp_storage = line_tmp_storage[: args.minibatch_size]
        importances_batch, bounds_storage, coverage_storage, _ = compute_batch_ranking(
            line_tmp_storage,
            numeric_column_types,
            args,
            cpu_pool,
            column_descriptions,
            logger,
            local_pbar,
        )

        for k, v in coverage_storage.items():
            local_coverage_object[k].append(v)

        step_timing_checkpoints.append(importances_batch.step_times)
        importances_df += importances_batch.triplet_scores
        bounds_storage = dict()
        bounds_storage_batch.append(bounds_storage)
        checkpoint_importances_df(importances_df)

    local_pbar.set_description('Wrapping up')
    local_pbar.close()

    return (
        step_timing_checkpoints,
        get_grouped_df(importances_df),
        GLOBAL_CARDINALITY_STORAGE,
        bounds_storage_batch,
        memory_storage_batch,
        local_coverage_object,
        GLOBAL_RARE_VALUE_STORAGE,
    )
