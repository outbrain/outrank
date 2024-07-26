from __future__ import annotations

from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import qr
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.utils import resample


class CategoricalClassification:

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.dataset_info = {
            'general': {},
            'combinations': [],
            'correlations': [],
            'duplicates': [],
            'labels': {},
            'noise': [],
        }

    def __repr__(self):
        return f"CategoricalClassification(dataset_info={self.dataset_info})"

    def generate_data(
        self,
        n_features: int,
        n_samples: int,
        cardinality: int = 5,
        structure: list | ArrayLike | None = None,
        ensure_rep: bool = False,
        random_values: bool | None = False,
        low: int | None = 0,
        high: int | None = 1000,
        k: int | float = 10,
        seed: int = 42,
    ) -> np.ndarray:

        """
        Generates dataset based on given parameters
        :param n_features: number of generated features
        :param n_samples: number of generated samples
        :param cardinality: default cardinality of the dataset
        :param structure: structure of the dataset
        :param ensure_rep: flag, ensures all given values represented
        :param random_values: flag, enables random (integer) feature values from set [low, high]
        :param low: sets lower bound of random feature values
        :param high: sets high bound of random feature values
        :param k: scale constant for normal distribution, default 10, sets width of normal distribution, larger value -> narrower peak
        :param seed: sets seed of numpy random
        :return: X, 2D dataset
        """

        self.dataset_info.update({
            'general': {
                'n_features': n_features,
                'n_samples': n_samples,
                'cardinality': cardinality,
                'structure': structure,
                'ensure_rep': ensure_rep,
                'seed': seed,
            },
        })

        np.random.seed(seed)
        X = np.empty([n_features, n_samples], dtype='int32')

        # No specific structure parameter passed
        if structure is None:
            for i in range(n_features):
                x = self._generate_feature(
                    n_samples,
                    cardinality=cardinality,
                    ensure_rep=ensure_rep,
                    random_values=random_values,
                    low=low,
                    high=high,
                    k=k,
                )
                X[i] = x
        # Structure parameter passed, building based on structure
        else:
            ix = 0
            for data in structure:

                # Data in structure is a tuple of (feature index (integer), feature attributes)
                if not isinstance(data[0], (list, np.ndarray)):
                    feature_ix, feature_attributes = data

                    # Filling out the dataset up to column index feature_ix
                    if ix < feature_ix:
                        for i in range(ix, feature_ix):
                            x = self._generate_feature(
                                n_samples,
                                cardinality=cardinality,
                                ensure_rep=ensure_rep,
                                random_values=random_values,
                                low=low,
                                high=high,
                                k=k,
                            )
                            X[ix] = x
                            ix += 1

                    x = self._configure_generate_feature(
                        feature_attributes,
                        n_samples,
                        ensure_rep=ensure_rep,
                        random_values=random_values,
                        low=low,
                        high=high,
                    )
                    X[ix] = x
                    ix += 1

                # Data in structure is a tuple of (list of feature indexes, feature attributes)
                else:
                    feature_ixs, feature_attributes = data

                    # Filling out the dataset up to feature_ix
                    for feature_ix in feature_ixs:
                        if ix < feature_ix:
                            for i in range(ix, feature_ix):
                                x = self._generate_feature(
                                    n_samples,
                                    cardinality=cardinality,
                                    ensure_rep=ensure_rep,
                                    random_values=random_values,
                                    low=low,
                                    high=high,
                                    k=k,
                                )
                                X[ix] = x
                                ix += 1

                        x = self._configure_generate_feature(
                            feature_attributes,
                            n_samples,
                            ensure_rep=ensure_rep,
                            random_values=random_values,
                            low=low,
                            high=high,
                        )

                        X[ix] = x
                        ix += 1

            # Fill out the rest of the dataset
            if ix < n_features:
                for i in range(ix, n_features):
                    x = self._generate_feature(
                        n_samples,
                        cardinality=cardinality,
                        ensure_rep=ensure_rep,
                        random_values=random_values,
                        low=low,
                        high=high,
                        k=k,
                    )
                    X[i] = x

        return X.T

    def _configure_generate_feature(
        self,
        feature_attributes: int | list | ArrayLike,
        n_samples: int,
        ensure_rep: bool = False,
        random_values: bool | None = False,
        low: int | None = 0,
        high: int | None = 1000,
        k: int | float = 10,
    ) -> np.ndarray:

        """
        Helper function, calls _generate_feature with appropriate parameters based on feature_attributes
        :param feature_attributes: either integer (cardinality) or list of feature attributes
        :param n_samples: number of samples in dataset
        :param ensure_rep: ensures all values are represented at least once in the feature vector
        :param random_values: randomly picked values for vec if true, otherwise values range from [low, cardinality] with by 1
        :param low: lower bound of random feature vector values
        :param high: upper bound of random feature vector values
        :param k: scale constant for normal distribution, default 10, sets width of normal distribution, larger value -> narrower peak
        :return: feature vector
        """

        # feature_cardinality is just an integer, generate feature either with random values or
        # [low, low+cardinality]
        if not isinstance(feature_attributes, (list, np.ndarray)):
            x = self._generate_feature(
                n_samples,
                cardinality=feature_attributes,
                ensure_rep=ensure_rep,
                random_values=random_values,
                low=low,
                high=high,
                k=k,
            )
        # feature_cardinality is a list of [value_domain, value_frequencies]
        else:
            if isinstance(feature_attributes[0], (list, np.ndarray)):
                value_domain, value_frequencies = feature_attributes
                x = self._generate_feature(
                    n_samples,
                    vec=value_domain,
                    ensure_rep=ensure_rep,
                    p=value_frequencies,
                )
            # feature_cardinality is value_domain (list of values for feature)
            else:
                value_domain = feature_attributes
                x = self._generate_feature(
                    n_samples,
                    vec=value_domain,
                    ensure_rep=ensure_rep,
                    k=k,
                )

        return x

    def _generate_feature(
        self,
        size: int,
        vec: list[int] | ArrayLike | None = None,
        cardinality: int = 5,
        ensure_rep: bool = False,
        random_values: bool | None = False,
        low: int | None = 0,
        high: int | None = 1000,
        p: list[float] | np.ndarray | None = None,
        k: int | float = 10,
    ) -> np.ndarray:
        """
        Generates feature vector of length size. Default probability density distribution is approximately normal, centred around a randomly picked value.
        :param vec: list of feature values
        :param cardinality: single value cardinality
        :param size: length of feature vector
        :param ensure_rep: ensures all values are represented at least once in the feature vector
        :param random_values: randomly picked values for vec if true, otherwise values range from [low, cardinality] with by 1
        :param low: lower bound of random feature vector values
        :param high: upper bound of random feature vector values
        :param p: list of probabilities of each value
        :param k: scale constant for normal distribution, default 10, sets width of normal distribution, larger value -> narrower peak
        :return: feature vector x
        """

        if vec is None:
            if random_values:
                vec = range(low, high + 1)
                vec = np.random.choice(vec, size=cardinality, replace=False)
            else:
                vec = np.arange(low, low + cardinality, 1)
        else:
            vec = np.array(vec)

        vec_len = len(vec)
        if p is None:
            v_shift = vec - vec[np.random.randint(len(vec))]
            p = norm.pdf(v_shift, scale=vec_len/k)
        else:
            p = np.array(p)

        p = p / p.sum()

        if ensure_rep and len(vec) < size:
            sampled_values = np.random.choice(vec, size=(size - len(vec)), p=p)
            sampled_values = np.append(sampled_values, vec)
        else:
            sampled_values = np.random.choice(vec, size=size, p=p)

        np.random.shuffle(sampled_values)
        return sampled_values.astype('int32')

    def generate_combinations(
        self,
        X: ArrayLike,
        feature_indices: list[int] | ArrayLike,
        combination_function: Optional = None,
        combination_type: Literal['linear', 'nonlinear'] = 'linear',
    ) -> np.ndarray:
        """
        Generates linear, nonlinear, or custom combinations within feature vectors in given dataset X
        :param X: dataset
        :param feature_indices: indexes of features to be in combination
        :param combination_function: optional custom function for combining feature vectors
        :param combination_type: string flag, either liner or nonlinear, defining combination type
        :return: X with added resultant feature
        """

        selected_features = X[:, feature_indices]

        if combination_function is None:
            if combination_type == 'linear':
                combination_function = lambda x: np.sum(x, axis=1)
            elif combination_type == 'nonlinear':
                combination_function = lambda x: np.sin(np.sum(x, axis=1))
        else:
            combination_type = str(combination_function.__name__)

        combination_result = combination_function(selected_features)

        combination_ix = len(X[0])

        self.dataset_info['combinations'].append({
            'feature_indices': feature_indices,
            'combination_type': combination_type,
            'combination_ix': combination_ix,
        })

        return np.column_stack((X, combination_result))

    def _xor(self, arr: list[int] | ArrayLike) -> np.ndarray:
        """
        Performs bitwise XOR operation on two integer arrays
        :param arr: features to perform XOR operation on
        :return: bitwise XOR result
        """
        arr = np.array(arr)
        arrT = arr.T
        arrT = arrT.astype(int)
        out = np.bitwise_xor(arrT[0], arrT[1])
        if len(arrT) > 2:
            for i in range(2, len(arrT)):
                out = np.bitwise_xor(out, arrT[i])

        return out.T

    def _and(self, arr: list[int] | ArrayLike) -> np.ndarray:
        """
        Performs bitwise AND operation on two integer arrays
        :param arr: features to perform AND operation on
        :return: bitwise AND result
        """
        arr = np.array(arr)
        arrT = arr.T
        arrT = arrT.astype(int)
        out = np.bitwise_and(arrT[0], arrT[1])
        if len(arrT) > 2:
            for i in range(2, len(arrT)):
                out = np.bitwise_and(out, arrT[i])

        return out.T

    def _or(self, arr: list[int] | ArrayLike) -> np.ndarray:
        """
        Performs bitwise OR operation on two integer arrays
        :param arr: features to perform OR operation on
        :return: bitwise OR result
        """
        arr = np.array(arr)
        arrT = arr.T
        arrT = arrT.astype(int)
        out = np.bitwise_or(arrT[0], arrT[1])
        if len(arrT) > 2:
            for i in range(2, len(arrT)):
                out = np.bitwise_or(out, arrT[i])

        return out.T

    def generate_correlated(
        self,
        X: ArrayLike,
        feature_indices: list[int] | ArrayLike,
        r: float = 0.8,
    ) -> np.ndarray:

        """
        Generates correlated features using the given feature indices. Correlation is based on cosine of angle between vectors with mean 0.
        :param X: dataset
        :param feature_indices: indices of features to generate correlated feature to
        :param r: (Pearson) correlation factor
        :return: X with generated correlated  features
        """

        if not isinstance(feature_indices, (list, np.ndarray)):
            feature_indices = np.array([feature_indices])

        if len(feature_indices) > 1:
            correlated_ixs = np.arange(len(X[0]), (len(X[0]) + len(feature_indices)), 1)
        else:
            correlated_ixs = len(X[0])

        selected_features = X[:, feature_indices]
        transposed = np.transpose(selected_features)
        correlated_features = []

        for t in transposed:
            theta = np.arccos(r)
            t_standard = (t - np.mean(t)) / (np.std(t) + 1e-10)

            rand = np.random.normal(0, 1, len(t_standard))
            rand = (rand - np.mean(rand)) / (np.std(rand) + 1e-10)

            M = np.column_stack((t_standard, rand))
            M_centred = (M - np.mean(M, axis=0))

            Id = np.eye(len(t))
            Q = qr(M_centred[:, [0]], mode='economic')[0]
            P = np.dot(Q, Q.T)
            orthogonal_projection = np.dot(Id - P, M_centred[:, 1])
            M_orthogonal = np.column_stack((M_centred[:, 0], orthogonal_projection))

            Y = np.dot(M_orthogonal, np.diag(1 / np.sqrt(np.sum(M_orthogonal ** 2, axis=0))))
            corr = Y[:, 1] + (1 / np.tan(theta)) * Y[:, 0]

            correlated_features.append(corr)

        correlated_features = np.transpose(correlated_features)

        self.dataset_info['correlations'].append({
            'feature_indices': feature_indices,
            'correlated_indices': correlated_ixs,
            'correlation_factor': r,
        })

        return np.column_stack((X, correlated_features))

    def generate_duplicates(
        self,
        X: ArrayLike,
        feature_indices: list[int] | ArrayLike,
    ) -> np.ndarray:
        """
        Generates duplicate features
        :param X: dataset
        :param feature_indices: indices of features to duplicate
        :return: dataset with duplicated features
        """
        if not isinstance(feature_indices, (list, np.ndarray)):
            feature_indices = np.array([feature_indices])

        duplicated_ixs = np.arange(len(X[0]), (len(X[0]) + len(feature_indices) - 1), 1)

        selected_features = X[:, feature_indices]

        self.dataset_info['duplicates'].append({
            'feature_indices': feature_indices,
            'duplicate_indices': duplicated_ixs,
        })

        return np.column_stack((X, selected_features))

    def generate_labels(
        self,
        X: ArrayLike,
        n: int = 2,
        p: float | list[float] | ArrayLike = 0.5,
        k: int | float = 2,
        decision_function: Optional = None,
        class_relation: Literal['linear', 'nonlinear', 'cluster'] = 'linear',
        balance: bool = False,
        random_state: int = 42,
    ):
        """
        Generates labels for dataset X
        :param X: dataset
        :param n: number of class labels
        :param p: class distribution
        :param k: constant
        :param decision_function: optional user-defined decision function
        :param class_relation: string, either 'linear', 'nonlinear', or 'cluster'
        :param balance: boolean, whether to balance clustering class labels
        :param random_state: seed for KMeans clustering, defaults to 42
        :return: array of labels, corresponding to dataset X
        """

        if isinstance(p, (list, np.ndarray)):
            if sum(p) > 1: raise ValueError('sum of values in must be less than 1.0')
            if len(p) > n: raise ValueError('length of p must equal n')
        elif p > 1.0: raise ValueError('p must be less than 1.0')

        n_samples, n_features = X.shape

        if decision_function is None:
            if class_relation == 'linear':
                decision_function = lambda x: np.sum(2 * x + 3, axis=1)
            elif class_relation == 'nonlinear':
                decision_function = lambda x: np.sum(k * np.sin(x) + k * np.cos(x), axis=1)
            elif class_relation == 'cluster':
                decision_function = None
        else:
            class_relation = str(decision_function.__name__)

        y = []
        if decision_function is not None:
            if n > 2:
                if type(p) != list:
                    p = 1 / n
                    percentiles = [p * 100]
                    for i in range(1, n - 1):
                        percentiles.append(percentiles[i - 1] + (p * 100))

                    decision_boundary = decision_function(X)
                    p_points = np.percentile(decision_boundary, percentiles)

                    y = np.zeros_like(decision_boundary, dtype=int)
                    for p_point in p_points:
                        y += (decision_boundary > p_point)
                else:
                    decision_boundary = decision_function(X)
                    percentiles = [x * 100 for x in p]

                    for i in range(1, len(percentiles) - 1):
                        percentiles[i] += percentiles[i - 1]
                    percentiles.insert(0, 0)
                    percentiles.pop()

                    p_points = np.percentile(decision_boundary, percentiles)

                    y = np.zeros_like(decision_boundary, dtype=int)

                    for i in range(1, n):
                        p_point = p_points[i]
                        y += np.where(decision_boundary > p_point, 1, 0)
            else:
                decision_boundary = decision_function(X)
                if isinstance(p, (list, np.ndarray)):
                    p = p[0]
                p_point = np.percentile(decision_boundary, p * 100)
                y = np.where(decision_boundary > p_point, 1, 0)
        else:
            if p == 0.5:
                p = 1.0
            else:
                p = [p, 1 - p]
            y = self._cluster_data(X, n, p=p, balance=balance, random_state=random_state)

        self.dataset_info.update({
            'labels': {
                'class_relation': class_relation,
                'n_class': n,
            },
        })

        return y

    def _cluster_data(
        self,
        X: ArrayLike,
        n: int,
        p: float | list[float] | ArrayLike | None = 1.0,
        balance: bool = False,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Cluster data using kmeans
        :param X: dataset
        :param n: number of clusters
        :param p: class distribution
        :param balance: balance the clusters according to p
        :random_state: seed for KMeans clustering, defaults to 42
        :return: array of labels, corresponding to dataset X
        """

        kmeans = KMeans(n_clusters=n, random_state=random_state)

        kmeans.fit(X)

        cluster_labels = kmeans.labels_

        # Fully balanced clusters
        if not isinstance(p, (list, np.ndarray)):
            samples_per_cluster = [len(X) // n] * n
        else:
            samples = len(X)
            samples_per_cluster = []
            if not isinstance(p, (list, np.ndarray)):
                samples_per_cluster.append(int(samples * p) // n)
                samples_per_cluster.append(int(samples * (1 - p)) // n)
            else:
                if len(p) == n:
                    for val in p:
                        samples_per_cluster.append(int(samples * val))
                else:
                    raise Exception('Length of balance parameter must equal number of clusters.')

        # Adjust cluster sizes
        if balance:
            adjustments = []
            overflow_samples = []
            overflow_indices = []
            for i in range(n):
                cluster_size = np.sum(cluster_labels == i)

                adjustment = samples_per_cluster[i] - cluster_size
                adjustments.append(adjustment)

                # Cluster is too large
                if adjustment < 0:
                    centroid = kmeans.cluster_centers_[i]
                    # Indices of samples in dataset
                    dataset_indices = np.where(cluster_labels == i)[0]
                    cluster_samples = np.copy(X[dataset_indices])

                    distances = np.linalg.norm(
                        cluster_samples - centroid,
                        axis=1,
                    )  # Distances of cluster samples to cluster centroid
                    cluster_sample_indices = np.argsort(distances)
                    dataset_indices_sorted = dataset_indices[
                        cluster_sample_indices
                    ]  # Indices of samples sorted by sample distance to cluster centroid

                    overflow_sample_indices = cluster_sample_indices[samples_per_cluster[i]:]  # Overflow samples
                    dataset_indices_sorted = dataset_indices_sorted[
                                             samples_per_cluster[i]:
                    ]  # Dataset indices of overflow samples

                    for i in range(len(overflow_sample_indices)):
                        overflow_samples.append(cluster_samples[overflow_sample_indices[i]])
                        overflow_indices.append(dataset_indices_sorted[i])

            overflow_samples = np.array(overflow_samples)
            overflow_indices = np.array(overflow_indices)

            # Making adjustments
            for i in range(n):

                if adjustments[i] > 0:
                    centroid = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(overflow_samples - centroid, axis=1)

                    closest_sample_indices = np.argsort(distances)

                    overflow_indices_sorted = overflow_indices[closest_sample_indices]

                    sample_indices_slice = closest_sample_indices[:adjustments[i]]
                    overflow_indices_slice = overflow_indices_sorted[:adjustments[i]]

                    cluster_labels[overflow_indices_slice] = i

                    overflow_samples = np.delete(overflow_samples, sample_indices_slice, axis=0)
                    overflow_indices = np.delete(overflow_indices, sample_indices_slice, axis=0)

        return np.array(cluster_labels)

    def generate_noise(
        self,
        X: ArrayLike,
        y: list[int] | ArrayLike,
        p: float = 0.2,
        type: Literal['categorical', 'missing'] = 'categorical',
        missing_val: str | int | float = float('-inf'),
    ) -> np.ndarray:

        """
        Simulates noise on given dataset X
        :param X: dataset to apply noise to
        :param y: required target labels for categorical noise generation
        :param p: amount of noise to apply. Defaults to 0.2
        :param type: type of noise to apply, either categorical or missing
        :param missing_val: value to simulate missing values. Defaults to float('-inf')
        :return: X with noise applied
        """

        self.dataset_info['noise'].append({
            'type': type,
            'amount': p,
        })

        if type == 'categorical':
            label_values, label_count = np.unique(y, return_counts=True)
            n_labels = len(label_values)

            inds = y.argsort()
            y_sort = y[inds]
            X_sort = X[inds]

            Xs_T = X_sort.T
            n = Xs_T.shape[1]
            n_flip = int(n * p)

            for feature in Xs_T:
                unique_per_label = {}

                for i in range(n_labels):
                    if i == 0:
                        unique = np.unique(feature[:label_count[i]])
                        unique_per_label[label_values[i]] = set(unique)
                    else:
                        unique = np.unique(feature[label_count[i - 1]:label_count[i - 1] + label_count[i] - 1])
                        unique_per_label[label_values[i]] = set(unique)

                ixs = np.random.choice(n, n_flip, replace=False)

                for ix in ixs:
                    current_label = y_sort[ix]
                    possible_labels = np.where(label_values != current_label)[0]

                    # find all unique values from labels != current label
                    values = set()
                    for key in possible_labels:
                        values = values.union(unique_per_label[key])

                    # remove any overlapping values, ensuring replacement values are unique & from a target label !=
                    # current label
                    for val in unique_per_label[current_label] & values:
                        values.remove(val)

                    if len(values) > 0:
                        val = np.random.choice(list(values))

                    else:
                        key = possible_labels[np.random.randint(len(possible_labels))]
                        values = unique_per_label[key]
                        val = np.random.choice(list(values))

                    feature[ix] = val

            rev_ind = inds.argsort()
            X_noise = Xs_T.T
            X_noise = X_noise[rev_ind]

            return X_noise

        elif type == 'missing':
            X_noise = np.copy(X)
            Xn_T = X_noise.T
            n = Xn_T.shape[1]
            n_missing = int(n * p)
            #print("n to delete:", n_missing)

            for feature in Xn_T:
                ixs = np.random.choice(n, n_missing, replace=False)

                for ix in ixs:
                    feature[ix] = missing_val

            return Xn_T.T

        else:
            raise ValueError(f'Type {type} not supported')

    def downsample_dataset(
        self,
        X: ArrayLike,
        y: list[int] | ArrayLike,
        n: int | None = None,
        seed: int = 42,
        reshuffle: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:

        """
        Downsamples dataset X according to N or the number of samples in minority class, resulting in a balanced dataset.
        :param X: Dataset to downsample
        :param y: Labels corresponding to X
        :param n: Optional number of samples per class to downsample to
        :param seed: Seed for random state of resample function
        :param reshuffle: Reshuffle the dataset after downsampling
        :return: Balanced X and y after downsampling
        """

        original_shape = X.shape

        values, counts = np.unique(y, return_counts=True)
        if n is None:
            n = min(counts)

        if n > min(counts):
            raise ValueError('N must be equal to or less than the number of samples in minority class')

        X_arrays_list = []
        y_downsampled = []
        for label in values:
            X_label = [X[i] for i in range(len(y)) if y[i] == label]
            X_label_downsample = resample(
                X_label,
                replace=True,
                n_samples=n,
                random_state=seed,
            )
            X_arrays_list.append(X_label_downsample)
            ys = [label] * n
            y_downsampled = np.concatenate((y_downsampled, ys), axis=0)

        X_downsampled = np.concatenate(X_arrays_list, axis=0)

        if reshuffle:
            indices = np.arange(len(X_downsampled))
            np.random.shuffle(indices)
            X_downsampled = X_downsampled[indices]
            y_downsampled = y_downsampled[indices]

        downsampled_shape = X_downsampled.shape

        self.dataset_info.update({
            'downsampling': {
                'original_shape': original_shape,
                'downsampled_shape': downsampled_shape,
            },
        })

        return X_downsampled, y_downsampled

    def print_dataset(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ):
        """
        Prints given dataset
        :param X: dataset
        :param y: labels
        :return:
        """

        n_samples, n_features = X.shape
        n = 0
        for arr in X:
            print('[', end='')
            for i in range(n_features):
                if i == n_features - 1:
                    print(arr[i], end='')
                else:
                    print(arr[i], end=', ')
            print(f'], Label: {y[n]}')
            n += 1

    # TODO: Logging function
