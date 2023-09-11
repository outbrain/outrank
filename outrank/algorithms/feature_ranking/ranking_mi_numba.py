from __future__ import annotations

import numpy as np
from numba import njit
from numba import prange

np.random.seed(123)
# Fast Numba-based approximative mutual information


@njit(
    'Tuple((int32[:], int32[:]))(int32[:])',
    cache=True,
    fastmath=True,
    error_model='numpy',
    boundscheck=True,
)
def numba_unique(a):
    """Identify unique elements in an array, fast"""

    len_a = a.shape[0]
    container = np.zeros(np.max(a) + 1, dtype=np.int32)
    for el in range(len_a):
        container[a[el]] += 1

    unique_values = np.where(container != 0)[0]
    unique_counts = container[unique_values]
    return unique_values.astype(np.int32), unique_counts.astype(np.int32)


@njit(
    'float32(int32[:], int32[:], int32, float32)',
    cache=True,
    fastmath=True,
    error_model='numpy',
    boundscheck=True,
)
def compute_conditional_entropy(Y_classes, class_values, class_var_shape, initial_prob):
    conditional_entropy = 0.0

    for c in class_values:
        conditional_prob = np.count_nonzero(Y_classes == c) / class_var_shape
        if conditional_prob != 0:
            conditional_entropy -= (
                initial_prob * conditional_prob * np.log(conditional_prob)
            )

    return conditional_entropy


@njit(
    'float32(int32[:], int32[:], int32, int32[:], int32[:], b1)',
    cache=True,
    parallel=False,
    fastmath=True,
    error_model='numpy',
    boundscheck=True,
)
def compute_entropies(
    X, Y, all_events, f_values, f_value_counts, cardinality_correction,
):
    """Core entropy computation function"""

    conditional_entropy = 0.0
    background_cond_entropy = 0.0
    full_entropy = 0.0

    class_values, class_counts = numba_unique(Y)

    if not cardinality_correction:
        for k in prange(len(class_counts)):
            class_probability = class_counts[k] / all_events
            full_entropy += -class_probability * np.log(class_probability)

    for f_index in prange(len(f_values)):
        _f_value_counts = f_value_counts[f_index]

        if _f_value_counts == 1:
            continue

        initial_prob = _f_value_counts / all_events
        x_value_subspace = np.where(X == f_values[f_index])
        Y_classes = Y[x_value_subspace]
        conditional_entropy += compute_conditional_entropy(
            Y_classes, class_values, _f_value_counts, initial_prob,
        )

        if cardinality_correction:
            # A neat hack that seems to work fine (permutations are expensive)
            Y_classes = np.roll(Y, _f_value_counts)[x_value_subspace]

            background_cond_entropy += compute_conditional_entropy(
                Y_classes, class_values, _f_value_counts, initial_prob,
            )

    if not cardinality_correction:
        return full_entropy - conditional_entropy

    else:
        # note: full entropy falls out during derivation of final term
        core_joint_entropy = -conditional_entropy + background_cond_entropy
        return core_joint_entropy


@njit(
    'float32(int32[:], int32[:], float32, b1)',
    cache=True,
    fastmath=True,
    error_model='numpy',
    boundscheck=True,
)
def mutual_info_estimator_numba(
    Y, X, approximation_factor=1, cardinality_correction=False,
):
    """Core estimator logic. Compute unique elements, subset if required"""

    all_events = len(X)
    f_values, f_value_counts = numba_unique(X)

    # Diagonal entries
    if np.sum(X - Y) == 0:
        cardinality_correction = False

    if approximation_factor < 1:
        subspace_size = int(approximation_factor * all_events)
        if subspace_size != 0:
            subspace = np.random.randint(0, all_events, size=subspace_size)
            X = X[subspace]
            Y = Y[subspace]

    joint_entropy_core = compute_entropies(
        X, Y, all_events, f_values, f_value_counts, cardinality_correction,
    )

    return approximation_factor * joint_entropy_core


if __name__ == '__main__':
    import pandas as pd
    from sklearn.feature_selection import mutual_info_classif

    np.random.seed(123)
    import time

    final_times = []
    for algo in ['MI-numba-randomized']:
        for order in range(20, 21):
            for j in range(1):
                start = time.time()
                a = np.random.randint(1000, size=2**order).astype(np.int32)
                b = np.random.randint(1000, size=2**order).astype(np.int32)
                if algo == 'MI':
                    final_score = mutual_info_classif(
                        a.reshape(-1, 1), b.reshape(-1), discrete_features=True,
                    )
                elif algo == 'MI-numba-randomized':
                    final_score = mutual_info_estimator_numba(
                        a, b, np.float32(1.0), True,
                    )
                elif algo == 'MI-numba':
                    final_score = mutual_info_estimator_numba(
                        a, b, np.float32(1.0), False,
                    )
                elif algo == 'MI-numba-randomized-ap':
                    final_score = mutual_info_estimator_numba(
                        a, b, np.float32(0.3), True,
                    )
                elif algo == 'MI-numba-ap':
                    final_score = mutual_info_estimator_numba(
                        a, b, np.float32(0.3), False,
                    )

                end = time.time()
                tdiff = end - start
                instance = {
                    'time': tdiff,
                    'samples 2e': order, 'algorithm': algo,
                }
                final_times.append(instance)
                print(instance)
    dfx = pd.DataFrame(final_times)
    dfx = dfx.sort_values(by=['samples 2e'])
    print(dfx)
