# Feature Evolution via Ranking

This script facilitates the process of feature evolution through iterative ranking using the `outrank` tool. It automates the process of running multiple iterations of feature ranking, extracting the best features, and updating the model specifications accordingly.

## Overview

The script performs the following steps:
1. **Initialization**: Sets up the initial model specification directory and creates the initial model JSON file.
2. **Iteration**: Runs the `outrank` task for a specified number of iterations.
3. **Feature Extraction**: Processes the results of each iteration to extract the best feature.
4. **Model Update**: Updates the model specification JSON with the newly identified best feature.

## Prerequisites

- Ensure that the `outrank` tool is installed and accessible from the command line.
- Python 3.6 or higher.
- Required Python packages: `pandas`, `argparse`, `json`, `shutil`, and `logging`.

## Installation

Install the required Python packages using pip (`pip install outrank --upgrade`)
