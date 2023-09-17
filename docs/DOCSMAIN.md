# Welcome to OutRank's documentation!

All functions/methods can be searched-for (search bar on the left).

This tool enables fast screening of feature-feature interactions. Its purpose is to give the user fast insight into potential redundancies/anomalies in the data.
It is implemented to operate in _mini batches_, it traverses the `raw data` incrementally, refining the rankings as it goes along. The core operation, interaction ranking, outputs triplets which look as follows:

```
featureA	featureB	0.512
featureA	featureC	0.125
```


# Use and installation - first steps (OutRank as a CLI)
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
* A minimal showcase of performing feature ranking on a generic CSV is demonstrated with [this example](../scripts/run_minimal.sh)

* [More examples](../scripts/) demonstrating OutRank's capabilities are also available.
