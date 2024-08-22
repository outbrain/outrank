from __future__ import annotations

import numpy as np
import numpy.typing as npt

np.random.seed(123)
max_size = 10**6


def max_pair_coverage(array1: npt.NDArray[np.int32], array2: npt.NDArray[np.int32]) -> float:
    def hash_pair(el1: np.int32, el2: np.int32):
        return (el1 * 1471343 - el2) % max_size

    counts = np.zeros(max_size, dtype=np.int32)
    tot_len = len(array1)
    for i in range(tot_len):
        identifier = hash_pair(array1[i], array2[i])
        counts[identifier] += 1

    return np.max(counts) / tot_len


if __name__ == '__main__':

    array1 = np.array([1,1,2,3,1,1,1,5] * 100000)
    array2 = np.array([0,0,5,5,3,0,0,0] * 100000)
    coverage = max_pair_coverage(array1, array2)
    assert coverage == 0.5
