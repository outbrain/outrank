from __future__ import annotations

import logging
import operator
import traceback
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

from outrank.algorithms.feature_ranking import ranking_cov_alignment
from outrank.core_utils import is_prior_heuristic

logger = logging.getLogger('syn-logger')
logger.setLevel(logging.DEBUG)

num_folds = 2

try:
    from outrank.algorithms.feature_ranking import ranking_mi_numba
    numba_available = True
except ImportError:
    traceback.print_exc()
    numba_available = False

def sklearn_MI(vector_first: np.ndarray, vector_second: np.ndarray) -> float:
    return mutual_info_classif(
        vector_first.reshape(-1, 1), vector_second.reshape(-1), discrete_features=True,
    )[0]

def sklearn_surrogate(
    vector_first: np.ndarray, vector_second: np.ndarray, X: np.ndarray, surrogate_model: str
        , is_target: bool=False,
) -> float:
    clf = initialize_classifier(surrogate_model)
    transf = OneHotEncoder()

    if not is_target:
        return 1.0

    if len(np.unique(vector_second)) > 2:
        vector_first, vector_second = vector_second, vector_first

    if X.size <= 1:
        X = vector_first.reshape(-1, 1)
    else:
        X = np.concatenate((X, vector_first.reshape(-1, 1)), axis=1)

    X = transf.fit_transform(X)

    scores = cross_val_score(clf, X, vector_second, scoring='neg_log_loss', cv=num_folds)
    return 1 + np.median(scores)

def numba_mi(vector_first: np.ndarray, vector_second: np.ndarray, heuristic: str, mi_stratified_sampling_ratio: float) -> float:
    cardinality_correction = heuristic == 'MI-numba-randomized'

    if vector_first.size == 2:
        if vector_first.shape[1] == 1:
            vector_first = vector_first.reshape(-1)
        else:
            vector_first = np.apply_along_axis(lambda x: np.abs(np.max(x) - np.sum(x)), 1, vector_first).reshape(-1)

    return ranking_mi_numba.mutual_info_estimator_numba(
        vector_first.astype(np.int32),
        vector_second.astype(np.int32),
        approximation_factor=np.float32(mi_stratified_sampling_ratio),
        cardinality_correction=cardinality_correction,
    )

def sklearn_mi_adj(vector_first: np.ndarray, vector_second: np.ndarray) -> float:
    return adjusted_mutual_info_score(vector_first, vector_second)

def generate_data_for_ranking(combination: tuple[str, str], reference_model_features: list[str], args: Any, tmp_df: pd.DataFrame) -> tuple(np.ndarray, np.ndrray):
    feature_one, feature_two = combination

    if feature_one == args.label_column:
        feature_one = feature_two
        feature_two = args.label_column

    if args.reference_model_JSON != '' and args.reference_model_JSON is not None:
        vector_first = tmp_df[list(reference_model_features) + [feature_one]].values
    else:
        vector_first = tmp_df[feature_one].values

    vector_second = tmp_df[feature_two].values
    return vector_first, vector_second


def conduct_feature_ranking(vector_first: np.ndarray, vector_second: np.ndarray, args: Any) -> float:

    heuristic = args.heuristic
    score = 0.0

    if heuristic == 'MI':
        score = sklearn_MI(vector_first, vector_second)

    elif heuristic in {'surrogate-SGD', 'surrogate-SVM', 'surrogate-SGD-prior'}:
        logger.warning('surrogate-based models currently not available .. Try a MI-based one (e.g., MI-numba-randomized).')
        score = 0.0

    elif heuristic == 'max-value-coverage':
        score = ranking_cov_alignment.max_pair_coverage(vector_first, vector_second)

    elif heuristic == 'MI-numba-randomized':
        score = numba_mi(vector_first, vector_second, heuristic, args.mi_stratified_sampling_ratio)

    elif heuristic == 'AMI':
        score = sklearn_mi_adj(vector_first, vector_second)

    elif heuristic == 'correlation-Pearson':
        score = pearsonr(vector_first, vector_second)[0]

    elif heuristic == 'Constant':
        score = 0.0

    else:
        logger.warning(f'{heuristic} not defined!')
        score = 0.0

    return score

def get_importances_estimate_pairwise(combination: tuple[str, str], reference_model_features: list[str], args: Any, tmp_df: pd.DataFrame) -> tuple[str, str, float]:

    feature_one, feature_two = combination
    inputs_encoded, output_encoded = generate_data_for_ranking(combination, reference_model_features, args, tmp_df)

    ranking_score = conduct_feature_ranking(inputs_encoded, output_encoded, args)

    return feature_one, feature_two, ranking_score


def rank_features_3MR(
    relevance_dict: dict[str, float],
    redundancy_dict: dict[tuple[Any, Any], Any],
    relational_dict: dict[tuple[Any, Any], Any],
    strategy: str = 'median',
    alpha: float = 1.0,
    beta: float = 1.0,
) -> pd.DataFrame:
    all_features = set(relevance_dict.keys())
    most_important_feature = max(relevance_dict.items(), key=operator.itemgetter(1))[0]
    ranked_features = [most_important_feature]

    def calc_higher_order(feature: str, is_redundancy: bool = True) -> float:
        values = []
        for feat in ranked_features:
            interaction_tuple = (feat, feature)
            if is_redundancy:
                values.append(redundancy_dict.get(interaction_tuple, 0))
            else:
                values.append(relational_dict.get(interaction_tuple, 0))
        return np.median(values) if strategy == 'median' else (np.mean(values) if strategy == 'mean' else sum(values))

    while len(ranked_features) < len(all_features):
        top_importance = -np.inf
        most_important_feature = None

        for feat in all_features - set(ranked_features):
            feature_redundancy = calc_higher_order(feat)
            feature_relation = calc_higher_order(feat, False)
            feature_relevance = relevance_dict[feat]
            importance = feature_relevance - alpha * feature_redundancy + beta * feature_relation

            if importance > top_importance:
                top_importance = importance
                most_important_feature = feat

        ranked_features.append(most_important_feature)

    return pd.DataFrame({'Feature': ranked_features, '3MR_Ranking': range(1, len(ranked_features) + 1)})

def get_importances_estimate_nonmyopic(args: Any, tmp_df: pd.DataFrame):
    pass

def initialize_classifier(surrogate_model: str):
    if 'surrogate-LR' in surrogate_model:
        return LogisticRegression(max_iter=100000)
    elif 'surrogate-SVM' in surrogate_model:
        return SVC(gamma='auto', probability=True)
    elif 'surrogate-SGD' in surrogate_model:
        return SGDClassifier(max_iter=100000, loss='log_loss')
    else:
        logger.warning(f'The chosen surrogate model {surrogate_model} is not supported, falling back to surrogate-SGD')
        return SGDClassifier(max_iter=100000, loss='log_loss')
