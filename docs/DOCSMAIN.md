# Welcome to OutRank's documentation!

All functions/methods can be searched-for (search bar on the left).

This tool enables fast screening of feature-feature interactions. Its purpose is to give the user fast insight into potential redundancies/anomalies in the data.
It is implemented to operate in _mini batches_, it traverses the `raw data` incrementally, refining the rankings as it goes along. The core operation, interaction ranking, outputs triplets which look as follows:

```
featureA	featureB	0.512
featureA	featureC	0.125
```


# Setup
```bash
pip install outrank
```

and test a minimal cycle with

```bash
outrank --task selftest
```

if this passes, you can be pretty certain OutRank will perform as intended. OutRank's primary use case is as a CLI tool, begin exploring with

```bash
outrank --help
```


# Example use cases
* A minimal showcase of performing feature ranking on a generic CSV is demonstrated with [this example](https://github.com/outbrain/outrank/tree/main/scripts/run_minimal.sh).

* [More examples](https://github.com/outbrain/outrank/tree/main/examples) demonstrating OutRank's capabilities are also available.


# OutRank as a Python library
Once installed, _OutRank_ can be used as any other Python library. For example, generic feature ranking algorithms can be accessed as

```python
from outrank.algorithms.feature_ranking.ranking_mi_numba import (
    mutual_info_estimator_numba,
)

# Some synthetic minimal data (Numpy vectors)
a = np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=np.int32)

lowest = np.array(np.random.permutation(a), dtype=np.int32)
medium = np.array([1, 1, 0, 0, 1, 1, 1, 1], dtype=np.int32)
high = np.array([1, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

lowest_score = mutual_info_estimator_numba(
	a, lowest, np.float32(1.0), False,
)
medium_score = mutual_info_estimator_numba(
	a, medium, np.float32(1.0), False,
)
high_score = mutual_info_estimator_numba(
	a, high, np.float32(1.0), False,
)

scores = [lowest_score, medium_score, high_score]
sorted_score_indices = np.argsort(scores)
assert np.sum(np.array([0, 1, 2]) - sorted_score_indices) ==  0
```
---
## Creating a simple dataset 
```python
from outrank.algorithms.synthetic_data_generators.cc_generator import CategoricalClassification

cc = CategoricalClassification()

# Creates a simple dataset of 10 features, 10k samples, with feature cardinality of all features being 35
X = cc.generate_data(9, 
                     10000, 
                     cardinality=35, 
                     ensure_rep=True, 
                     random_values=True, 
                     low=0, 
                     high=40)

# Creates target labels via clustering
y = cc.generate_labels(X, n=2, class_relation='cluster')

```