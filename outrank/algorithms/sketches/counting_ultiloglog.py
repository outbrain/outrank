"""
This module implements probabilistic data structure which is able to calculate the cardinality of large multisets in a single pass using little auxiliary memory
"""
from __future__ import annotations

import sys

import numpy as np
import xxhash


class HyperLogLogWCache:
    def __init__(self, max_prehash_size=1000000):
        # int(np.ceil(np.log2((1.04 / error_rate) ** 2)))
        self.p = 19
        self.m = 1 << self.p
        self.warmup_set = set()
        self.warmup_size = max_prehash_size
        self.width = 64 - self.p
        self.hll_flag = False

    def _hasher_update(self, value):
        self.hasher = xxhash.xxh32(seed=self.p)
        if isinstance(value, str):
            value = value.encode('utf-8')
            self.hasher.update(bytes(value))
        else:
            self.hasher.update(bytes(value))

        x = self.hasher.intdigest()
        j = x & (self.m - 1)
        w = x >> self.p

        rho = self.width - w.bit_length()
        self.M[j] = max(self.M[j], rho)

    def add(self, value):
        if sys.getsizeof(self.warmup_set) < self.warmup_size and not self.hll_flag:
            self.warmup_set.add(value)
        elif not self.hll_flag:
            if not self.hll_flag:
                self.M = np.zeros(self.m)
                for element in self.warmup_set:
                    self._hasher_update(element)
                self.warmup_set = {}
            self.hll_flag = True
        else:
            self._hasher_update(value)

    def __len__(self):
        if self.hll_flag:
            basis = np.ceil(
                self.m *
                np.log(np.divide(self.m, len(np.where(self.M == 0)[0]))),
            )
            if basis != np.inf:
                return int(basis) - 1
            else:
                return 2**self.p
        else:
            return len(self.warmup_set)


def cardinality_kernel(algo = 'cache'):

    start_time = time.time()

    if algo == 'Hhll (10)':
        GLOBAL_CARDINALITY_STORAGE = {1: None}
        GLOBAL_CARDINALITY_STORAGE[1] = HyperLogLogWCache(10)
    elif algo == 'Hhll (10000)':
        GLOBAL_CARDINALITY_STORAGE = {1: None}
        GLOBAL_CARDINALITY_STORAGE[1] = HyperLogLogWCache(10000)
    elif algo == 'hll+ (0.005)':
        import hyperloglog
        GLOBAL_CARDINALITY_STORAGE = {1: None}
        GLOBAL_CARDINALITY_STORAGE[1] = hyperloglog.HyperLogLog(0.005)
    elif algo == 'hll+ (0.01)':
        import hyperloglog
        GLOBAL_CARDINALITY_STORAGE = {1: None}
        GLOBAL_CARDINALITY_STORAGE[1] = hyperloglog.HyperLogLog(0.01)
    elif algo == 'set':
        GLOBAL_CARDINALITY_STORAGE = {1: set()}

    for j in ground:
        GLOBAL_CARDINALITY_STORAGE[1].add(j)

    size1 = asizeof.asizeof(GLOBAL_CARDINALITY_STORAGE)
    error1 = 100 * \
        (1 - len(GLOBAL_CARDINALITY_STORAGE[1]) / len(set(ground)))
    end_time = time.time()
    tp1 = end_time - start_time
    return tp1, error1


if __name__ == '__main__':
    import random
    import string
    import time

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import tqdm
    from pympler import asizeof
#    sns.set_style("whitegrid")
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'Helvetica',
    })
    def get_random_string(length):
        # choose from all lowercase letter
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str

    # results_df = []
    # num_vals = 100000
    # for _ in range(10):
    #     for j in tqdm.tqdm(range(1000, 100000, 1000)):
    #         ground = list(set(np.random.randint(0, j, num_vals).tolist()))
    #         ground = ground + [
    #             get_random_string(random.randint(1, 15)) for k in range(j)
    #         ]


    #         for algo in ['Hhll (10)', 'Hhll (10000)', 'hll+ (0.005)', 'hll+ (0.01)', 'set']:
    #             tp, error = cardinality_kernel(algo)
    #             results_df.append(
    #                 {
    #                     'num_samples': len(ground),
    #                     'time': tp,
    #                     'algo': algo,
    #                     'error': error,
    #                 }
    #             )


    # out_df = pd.DataFrame(results_df)
    # out_df.to_csv('backup.csv')
    pals = 'coolwarm'
    out_df = pd.read_csv('backup.csv')
    print(out_df)
    print(out_df.groupby('algo').mean())
    g = sns.jointplot(
        y=out_df.num_samples, x=out_df.error,
        hue=out_df.algo, alpha=0.6, palette=pals,
    )
    plt.tight_layout()
    g.ax_marg_y.remove()
    plt.ylim(0, max(out_df.num_samples.astype(float)))

    plt.ylabel('Num. of unique values in data')
    plt.xlabel('Abs error')
    plt.savefig('hllErr.pdf')
    plt.clf()
    plt.cla()


    sns.histplot(
        y=out_df.num_samples.astype(
            float,
        ), x=out_df.time, hue=out_df.algo,
        alpha=0.3,
        palette=pals,

    )
    g = sns.jointplot(
        y=out_df.num_samples.astype(
            float,
        ), x=out_df.time, hue=out_df.algo, alpha=.6, style=out_df.algo,
        palette=pals,
    )
    g.ax_marg_y.remove()
    plt.ylim(0, max(out_df.num_samples.astype(float)))
    plt.tight_layout()
    plt.xlabel('Num. of unique values in data')
    plt.ylabel('Time (s)')
    plt.savefig('hllTime.pdf')
    plt.clf()
    plt.cla()
