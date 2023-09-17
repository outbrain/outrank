from __future__ import annotations

import argparse
import json
import logging

from outrank.task_generators import outrank_task_generate_data_set
from outrank.task_ranking import outrank_task_conduct_ranking
from outrank.task_selftest import conduct_self_test
from outrank.task_summary import outrank_task_result_summary
from outrank.task_visualization import outrank_task_visualize_results

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logging.getLogger(__name__).setLevel(logging.INFO)

usage_examples = """
    Usage examples:

    # perform ranking, summary and visualize the results
    outrank --task all --data_path pathToSomeData --data_source ob-vw --heuristic MI-numba-randomized --include_cardinality_in_feature_names True --target_ranking_only True --combination_number_upper_bound 2048 --num_threads 8 --interaction_order 1 --transformers fw-transformers --output_folder ./ranking_outputs --subsampling 100

    # pairwise ranking only
    outrank --task ranking --data_path pathToSomeData --data_source ob-vw --heuristic MI-numba-randomized --target_ranking_only False --combination_number_upper_bound 10000 --num_threads 30 --output_folder ./ranking_outputs --subsampling 10

    # Higher order interactions
    outrank --task all --data_path pathToSomeData --data_source csv-raw --heuristic MI-numba-randomized --target_ranking_only True --combination_number_upper_bound 2048 --num_threads 8 --interaction_order 3 --output_folder ./ranking_outputs --subsampling 20

    # More docs and use cases at https://improved-dollop-9k1wgvm.pages.github.io/outrank.html
"""


def main():
    parser = argparse.ArgumentParser(
        description='Fast feature screening for sparse data sets.',
        epilog=usage_examples,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '--task',
        type=str,
        default='all',
        help='Type of task to consider. Can be either "ranking", "ranking_summary", "feature_summary_transformers",  or "visualization"',
    )

    parser.add_argument(
        '--minibatch_size',
        type=int,
        default=2**13,
        help='Suitable for data, not pre-split to batches, this parameter determines batch size - note that too large batch size can slow down the multithreaded score computation due to many thread allocations etc. This works ok for <300 features and up to 48 threads.',
    )

    parser.add_argument(
        '--output_folder',
        type=str,
        default='ranking_outputs',
        help='Output folder containing ranking results.',
    )

    parser.add_argument(
        '--data_source',
        type=str,
        default='ob-vw',
        help='Which database is used to obtain learning instances? this determines the inferred folder structure (csv-raw, ob-vw, ob-csv).',
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to the folder containing the main data used for subsequent learning.',
    )

    parser.add_argument(
        '--subsampling',
        type=int,
        default=10,
        help='Subsampling ratio - every n-th instance will be considered (suggested value: 10 to 100)',
    )

    parser.add_argument(
        '--combination_number_upper_bound',
        type=int,
        default=2**15,
        help='Cap the number of generated combinations (random sampling of that space).',
    )

    parser.add_argument(
        '--missing_value_symbols',
        type=str,
        default=',{}',
        help='What symbols denote missing values? Comma-separate them.',
    )

    parser.add_argument(
        '--heuristic',
        type=str,
        default='MI-numba-randomized',
        help='Selected heuristic (that performs feature scoring). For full list please see the docs: https://improved-dollop-9k1wgvm.pages.github.io/outrank/algorithms/importance_estimator.html#get_importances_estimate_pairwise',
    )

    parser.add_argument(
        '--include_noise_baseline_features',
        type=str,
        default='False',
        help='If enabled, it computes five control variables (random noises)',
    )

    parser.add_argument(
        '--include_cardinality_in_feature_names',
        type=str,
        default='True',
        help='If enabled, feature names appear as feature-(cardinality) for easier inspection/debugging.',
    )

    parser.add_argument(
        '--image_format',
        type=str,
        default='pdf',
        help='The format of the output images (task: visualization)',
    )

    parser.add_argument(
        '--num_threads', type=int, default=8, help='Number of threads to consider',
    )

    parser.add_argument(
        '--label_column',
        type=str,
        default='label',
        help='Name of the target attribute (useful for visualization)',
    )

    parser.add_argument(
        '--transformers',
        type=str,
        default='none',
        help='Collection of which feature transformations to consider',
    )

    parser.add_argument(
        '--rare_value_count_upper_bound',
        type=int,
        default=1,
        help="When identifying rare attr-val pairs, what's the upper frequency bound?",
    )

    parser.add_argument(
        '--feature_set_focus',
        type=str,
        default=None,
        help='Collection of which feature transformations to consider',
    )

    parser.add_argument(
        '--interaction_order',
        type=int,
        default=1,
        help='The order of feature interactions to consider during ranking (complex features comprised of n elementary ones)',
    )

    parser.add_argument(
        '--reference_model_JSON',
        type=str,
        default='./ranking_outputs/reference_model.json',
        help='Reference model JSON',
    )

    parser.add_argument(
        '--target_ranking_only',
        type=str,
        default='True',
        help='Compute only the feature-label scores? This is substantially faster (O(n)).',
    )

    parser.add_argument(
        '--explode_multivalue_features',
        type=str,
        default='False',
        help="Which ';'-separated features should be one-hot encoded into n new features (coverage analysis)",
    )

    parser.add_argument(
        '--subfeature_mapping',
        type=str,
        default='False',
        help='Compute sub-features on-the fly. Example: featureA->featureB implies features based on each value of featureA will be considered.',
    )

    parser.add_argument(
        '--num_synthetic_features',
        type=int,
        default=100,
        help='Relevant for task data_generator -- how many features.',
    )

    parser.add_argument(
        '--num_synthetic_rows',
        type=int,
        default=1000000,
        help='Relevant for task data_generator -- how many rows.',
    )

    parser.add_argument(
        '--generator_type',
        type=str,
        default='naive',
        help='Relevant for task data_generator -- which generator to consider',
    )

    parser.add_argument(
        '--output_synthetic_df_name',
        type=str,
        default='test_data_synthetic',
        help='Relevant for task data_generator -- name of the folder that contains generated data.',
    )

    args = parser.parse_args()

    if args.task == 'selftest':
        conduct_self_test()
        exit()

    if args.data_path is None and args.task != 'data_generator':
        logging.error('Please specify data set name (--data_path).')
        exit()

    all_tasks_to_consider = []
    if args.task != 'all':
        all_tasks_to_consider = [args.task]

    else:
        all_tasks_to_consider = ['ranking', 'ranking_summary', 'visualization']

    for task in all_tasks_to_consider:
        logging.info(f'Proceeding with task: {task} ..')

        if (
            task == 'ranking'
            or task == 'feature_summary_transformers'
            or task == 'identify_rare_values'
        ):
            outrank_task_conduct_ranking(args)

        elif task == 'visualization':
            outrank_task_visualize_results(args)

        elif task == 'ranking_summary':
            outrank_task_result_summary(args)

        elif task == 'data_generator':
            outrank_task_generate_data_set(args)

        else:
            logging.info(f'Warning, the selected task: {task} does not exist.')


if __name__ == '__main__':
    main()
