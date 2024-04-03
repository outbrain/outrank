# A module for pairwise computation of importances -- entrypoint for the core ranking algorighm(s)
from __future__ import annotations

import logging
import operator
import traceback
from typing import Any
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

from outrank.core_utils import is_prior_heuristic


logger = logging.getLogger('syn-logger')
logger.setLevel(logging.DEBUG)

num_folds = 4

try:
    from outrank.algorithms.feature_ranking import ranking_mi_numba

    numba_available = True

except Exception as es:
    traceback.print_exc(0)
    numba_available = False


def sklearn_MI(vector_first: Any, vector_second: Any) -> float:
    estimate_feature_importance = mutual_info_classif(
        vector_first.reshape(-1, 1), vector_second.reshape(-1), discrete_features=True,
    )[0]
    return estimate_feature_importance


def sklearn_surrogate(
    vector_first: Any, vector_second: Any, X: Any, surrogate_model: str
) -> float:
    
    clf = initialize_classifier(surrogate_model)
    
    transf = OneHotEncoder()

    # They do not commute, swap if needed
    if len(np.unique(vector_second) > 2):
        vector_third = vector_second
        vector_second = vector_first
        vector_first = vector_third
        del vector_third

    if X.size <= 1:
        X = vector_first.reshape(-1, 1)
    else:
        X = np.concatenate((X, vector_first.reshape(-1, 1)), axis=1)

    X = transf.fit_transform(X)
    estimate_feature_importance_list = cross_val_score(
        clf, X, vector_second, scoring='neg_log_loss', cv=num_folds,
    )
    estimate_feature_importance = 1 + \
        np.median(estimate_feature_importance_list)        

    return estimate_feature_importance


def numba_mi(vector_first, vector_second, heuristic, mi_stratified_sampling_ratio):
    if heuristic == 'MI-numba-randomized':
        cardinality_correction = True

    else:
        cardinality_correction = False

    estimate_feature_importance = ranking_mi_numba.mutual_info_estimator_numba(
        vector_first.reshape(-1).astype(np.int32),
        vector_second.reshape(-1).astype(np.int32),
        approximation_factor=np.float32(mi_stratified_sampling_ratio),
        cardinality_correction=cardinality_correction,
    )

    return estimate_feature_importance


def sklearn_mi_adj(vector_first, vector_second):
    # AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]
    estimate_feature_importance = adjusted_mutual_info_score(
        vector_first.reshape(-1), vector_second.reshape(-1),
    )
    return estimate_feature_importance


def get_importances_estimate_pairwise(combination, reference_model_features, args, tmp_df):
    """A method for parallel importances estimation. As interaction scoring is independent, individual scores can be computed in parallel."""

    feature_one = combination[0]
    feature_two = combination[1]

    if feature_one not in tmp_df.columns:
        logging.info(f'{feature_one} not found in the constructed data frame - consider increasing --combination_number_upper_bound for better coverage.')
        return [feature_one, feature_two, 0]
    elif feature_two not in tmp_df.columns:
        logging.info(f'{feature_two} not found in the constructed data frame - consider increasing --combination_number_upper_bound for better coverage.')
        return [feature_one, feature_two, 0]

    vector_first = tmp_df[[feature_one]].values.ravel()
    vector_second = tmp_df[[feature_two]].values.ravel()

    if len(vector_first) == 0 or len(vector_second) == 0:
        return [feature_one, feature_two, 0]

    # Compute score based on the selected heuristic.
    if args.heuristic == 'MI':
        # Compute the infoGain
        estimate_feature_importance = sklearn_MI(vector_first, vector_second)

    elif 'surrogate-' in args.heuristic:
        X = np.array(float)
        if is_prior_heuristic(args) and (len(reference_model_features) > 0):
            X = tmp_df[reference_model_features].values

        estimate_feature_importance = sklearn_surrogate(
            vector_first, vector_second, X, args.heuristic
        )

    elif 'MI-numba' in args.heuristic:
        estimate_feature_importance = numba_mi(
            vector_first, vector_second, args.heuristic, args.mi_stratified_sampling_ratio,
        )

    elif args.heuristic == 'AMI':
        estimate_feature_importance = sklearn_mi_adj(
            vector_first, vector_second,
        )

    elif args.heuristic == 'correlation-Pearson':
        estimate_feature_importance = pearsonr(vector_first, vector_second)[0]

    elif args.heuristic == 'Constant':
        estimate_feature_importance = 0.0

    else:
        raise ValueError(
            'Please select one of the possible heuristics (MI, chi2)',
        )

    return (feature_one, feature_two, estimate_feature_importance)


def rank_features_3MR(
    relevance_dict: dict[str, float],
    redundancy_dict: dict[tuple[Any, Any], Any],
    relational_dict: dict[tuple[Any, Any], Any],
    strategy: str = 'median',
    alpha: float = 1,
    beta: float = 1,
) -> pd.DataFrame:
    all_features = relevance_dict.keys()
    most_important_feature = max(
        relevance_dict.items(), key=operator.itemgetter(1),
    )[0]
    ranked_features = [most_important_feature]

    def calc_higher_order(feature, is_redundancy=True):
        values = []
        for feat in ranked_features:
            interaction_tuple = (feat, feature)
            if is_redundancy:
                if interaction_tuple in redundancy_dict:
                    values.append(redundancy_dict[interaction_tuple])
                else:
                    logging.info('Not accounting for redundancy tuple {} - please increase the --combination_number_upper_bound for beter coverage of interactions/redundancies.')
            else:
                if interaction_tuple in relational_dict:
                    values.append(relational_dict[interaction_tuple])
                else:
                    logging.info('Not accounting for interaction tuple {} - please increase the --combination_number_upper_bound for beter coverage of interactions/redundancies.')

        if strategy == 'sum':
            return sum(values)
        if strategy == 'mean':
            return np.mean(values)
        return np.median(values)

    while len(ranked_features) != len(all_features):
        top_importance = 0
        most_important_feature = ''

        for ind, feat in enumerate(set(all_features) - set(ranked_features)):
            feature_redundancy = calc_higher_order(feat)
            feature_relation = calc_higher_order(feat, False)
            feature_relevance = relevance_dict[feat]
            importance = (
                feature_relevance - alpha * feature_redundancy + beta * feature_relation
            )

            if (importance > top_importance) or (ind == 0):
                top_importance = importance
                most_important_feature = feat
        ranked_features.append(most_important_feature)
    return pd.DataFrame(
        {
            'Feature': ranked_features,
            '3mr_ranking': list(range(1, len(ranked_features) + 1)),
        },
    )


def get_importances_estimate_nonmyopic(args: Any, tmp_df: pd.DataFrame):
    # TODO - nonmyopic algorithms - tmp_df \ args.label vs. label
    # TODO - this is to be executed directly on df - no need for parallel kernel(s)
    pass


def initialize_classifier(surrogate_model: str):
    if 'surrogate-LR' in surrogate_model:
        return LogisticRegression(max_iter=100000)
    elif 'surrogate-SVM' in surrogate_model:
        return SVC(gamma='auto', probability=True)
    elif 'surrogate-SGD' in surrogate_model:
        return SGDClassifier(max_iter=100000, loss='log_loss')
    else:
        logging.warning(f'The chosen surrogate model {surrogate_model} is not supported, falling back to surrogate-SGD')
        return SGDClassifier(max_iter=100000, loss='log_loss')
