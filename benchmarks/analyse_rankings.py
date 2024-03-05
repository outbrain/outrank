from __future__ import annotations

import glob
import os
import sys

import matplotlib.pyplot as plt

def extract_just_ranking(dfile):
    """Extract ranking from an output file."""
    ranks = []
    with open(dfile) as df:
        next(df)  # Skip header line
        for line in df:
            parts = line.strip().split('\t')
            ranks.append(parts[1])
    return ranks

def calculate_mismatch_scores(all_folders, mismatches):
    """Calculate mismatch scores based on ranking files."""
    all_counts = [int(folder.split('_').pop()) for folder in all_folders if 'ranking' in folder]

    ranking_out_struct = {}
    for count in all_counts:
        rpath = os.path.join(dfolder, f'ranking_{count}', 'feature_singles.tsv')
        ranking_out_struct[count] = extract_just_ranking(rpath)

    pivot_score_key = max(all_counts)
    reference_ranking = ranking_out_struct[pivot_score_key]

    out_results = {}
    for ranking_id, ranking in ranking_out_struct.items():
        mismatches_counter = 0
        for el in ranking[:mismatches]:
            if el not in reference_ranking[:mismatches]:
                mismatches_counter += 1
        out_results[ranking_id] = 100 * (1 - mismatches_counter / mismatches)

    return dict(sorted(out_results.items(), key=lambda x: x[0]))

def plot_precision_curve(results, pivot_score_key, mismatches, axs, c1, c2):
    """Plot the precision curve based on mismatch results."""
    instances = [100 * (k / pivot_score_key) for k in results.keys()]
    values = list(results.values())

    axs[c1,c2].plot(instances, values, marker='o', linestyle='-', color='black')
    axs[c1,c2].invert_xaxis()
    axs[c1,c2].set(xlabel='Proportion of data used (%)', ylabel=f'hits@{mismatches} (%)', title=f'Approximation, top {mismatches} Features')
    axs[c1,c2].grid(True)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python script.py <directory>')
        sys.exit(1)

    dfolder = sys.argv[1]
    mismatch_range = [1, 5, 10, 20]
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    row = -1
    for enx, mismatches in enumerate(mismatch_range):
        if enx % 2 == 0:
            row += 1
        col = enx % 2
        all_folders = list(glob.glob(os.path.join(dfolder, '*')))
        out_results = calculate_mismatch_scores(all_folders, mismatches)
        pivot_score_key = max(out_results)
        plot_precision_curve(out_results, pivot_score_key, mismatches, axs, row, col)
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300)
