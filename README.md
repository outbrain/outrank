
    ░█████╗░██╗░░░██╗████████╗██████╗░░█████╗░███╗░░██╗██╗░░██╗
    ██╔══██╗██║░░░██║╚══██╔══╝██╔══██╗██╔══██╗████╗░██║██║░██╔╝
    ██║░░██║██║░░░██║░░░██║░░░██████╔╝███████║██╔██╗██║█████═╝░
    ██║░░██║██║░░░██║░░░██║░░░██╔══██╗██╔══██║██║╚████║██╔═██╗░
    ╚█████╔╝╚██████╔╝░░░██║░░░██║░░██║██║░░██║██║░╚███║██║░╚██╗
    ░╚════╝░░╚═════╝░░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░░╚═╝


[![CI - package](https://github.com/outbrain/outrank/actions/workflows/python-package.yml/badge.svg)](https://github.com/outbrain/outrank/actions/workflows/python-package.yml) [![CI - benchmark](https://github.com/outbrain/outrank/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/outbrain/outrank/actions/workflows/benchmarks.yml) [![CI - selftest](https://github.com/outbrain/outrank/actions/workflows/selftest.yml/badge.svg)](https://github.com/outbrain/outrank/actions/workflows/selftest.yml)
# Feature interaction module

This tool enables fast screening of feature-feature interactions. Its purpose is to give the user fast insight into potential redundancies/anomalies in the data.
It is implemented to operate in _mini batches_, it traverses the `raw data` incrementally, refining the rankings as it goes along.
The interaction ranking outputs triplets which look as follows:

```
featureA	featureB	0.512
featureA	featureC	0.125
```


# Use - CLI
```bash
pip install outrank
```

and test a minimal cycle with

```bash
outrank --task selftest
```

if this passes, you can be pretty certain OutRank will perform as intended.

OutRank's primary use case is as a CLI tool, begin exploring with

```bash
outrank --help
```

A minimal showcase is demonstrated with [this example](./run_minimal.sh)

# Contributing
1. Make sure the functionality is not already implemented!
2. Decide whether where the functionality would fit best (is it an algorithm? A parser?)
3. Open a PR with rationale


# Bugs and other reports
Feel free to open a PR that contains:
1. Issue overview
2. Minimal example useful for replicating the issue on our end
3. Possible solution
