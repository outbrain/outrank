from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from outrank.core_utils import read_reference_json

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
plt.rcParams['figure.figsize'] = (50, 30)


def visualize_hierarchical_clusters(
    triplet_dataframe: pd.DataFrame,
    output_folder: str,
    image_format: str = 'png',
    max_num_clusters: int = 100,
) -> None:
    """A method for visualization of hierarchical clusters w.r.t. different linkage functions"""

    # Prepare the canvas
    plt.rcParams['figure.figsize'] = (10, 5)
    unique_features = triplet_dataframe.FeatureA.unique()

    if len(unique_features) > 1000:
        logging.info('Trying to visualize too many features, exiting ..')
        exit()

    dmat = np.zeros((len(unique_features), len(unique_features)))
    logging.info('Preparing the data for clustering ..')

    if triplet_dataframe.shape[0] > 10**5:
        logging.info(
            'Trying to visualize more than 10 ** 5 triplets, exiting ..',
        )
        exit()

    pivot_table = pd.pivot_table(
        triplet_dataframe,
        values='Score',
        index='FeatureA',
        columns='FeatureB',
        aggfunc=np.mean,
    )

    # We need distances
    pivot_table.fillna(0, inplace=True)
    dmat = 1 - pivot_table.values

    # Visualize different dendrograms
    logging.info('Clustering ..')

    for linkage_heuristic in [
        # 'single', 'complete', 'average', 'weighted', 'centroid'
        'complete',
    ]:
        # Compute the linkage structure
        Z = hierarchy.linkage(dmat, linkage_heuristic)

        # Visualize
        hierarchy.dendrogram(
            Z, above_threshold_color='y', orientation='top', labels=unique_features,
        )
        # Store
        plt.title(f'Linkage function: {linkage_heuristic}')
        plt.tight_layout()
        out_path = f'{output_folder}/dendrogram_{linkage_heuristic}.{image_format}'
        plt.savefig(out_path, dpi=300)

        # Clean for subsequent plots
        plt.clf()
        plt.cla()
        logging.info(
            f'Visualized hierarchical clustering with linkage {linkage_heuristic} to {out_path}',
        )

        # Step 1: Identify relevant distance threshold bounds
        range_min, range_max = np.min(
            pivot_table.values,
        ), np.max(pivot_table.values)
        spectrum = np.arange(
            range_min, range_max,
            (range_max - range_min) / 1000,
        )
        max_silhouette = 0
        top_clustering = []
        full_silhouette_space = []

        # Step 2: Compute Silhouette for each threshold and store the results
        for possible_threshold in spectrum:
            cluster_assignments = hierarchy.fcluster(Z, possible_threshold)
            num_clusters = len(np.unique(cluster_assignments))
            if num_clusters > 2 and num_clusters < max_num_clusters:
                try:
                    sil_score = silhouette_score(
                        pivot_table, cluster_assignments,
                    )

                except Exception:
                    continue

                full_silhouette_space.append(
                    [
                        sil_score, possible_threshold, len(
                            np.unique(cluster_assignments),
                        ),
                    ],
                )
                if sil_score >= max_silhouette:
                    top_clustering = cluster_assignments
                    max_silhouette = sil_score

        # Step 3: We are interested in the best clustering w.r.t. Silhouette
        dfx = pd.DataFrame(full_silhouette_space)
        if len(dfx) == 0:
            logging.info('Silhouette space empty, exiting')
            exit()

        dfx.columns = ['Silhouette', 'threshold', 'numClusters']
        sns.lineplot(x=dfx.numClusters, y=dfx.Silhouette, color='black')
        plt.tight_layout()
        out_path = f'{output_folder}/SilhouetteProfile.{image_format}'
        plt.savefig(out_path, dpi=300)
        plt.clf()
        plt.cla()
        logging.info('Stored the Silhouette profile.')

        final_feature_cluster_df = pd.DataFrame(
            list(zip(top_clustering, pivot_table.index)),
        )
        final_feature_cluster_df.columns = ['ClusterID', 'Feature']
        final_feature_cluster_df.to_csv(
            f'{output_folder}/TopClustering.tsv', sep='\t',
        )

        # Get 2D embeddings of features and visualize them
        try:
            projected_data = TSNE().fit_transform(pivot_table.values)
            projected_data = pd.DataFrame(projected_data)
            projected_data.columns = ['Dim1', 'Dim2']
            projected_data['ClusterID'] = top_clustering
            projected_data['ClusterID'] = projected_data['ClusterID'].astype(
                str,
            )
            sns.scatterplot(
                x=projected_data.Dim1,
                y=projected_data.Dim2,
                hue=projected_data.ClusterID,
                palette='Set2',
            )
            plt.savefig(
                f'{output_folder}/clustersEmbeddingVisualization.pdf', dpi=300,
            )
            plt.clf()
            plt.cla()
        except:
            pass

        # Step 4: We are interested in the best clustering w.r.t. Silhouette
        # Not here yet

    plt.rcParams['figure.figsize'] = (50, 30)


def visualize_heatmap(
    triplets: pd.DataFrame, output_folder: str, image_format: str,
) -> None:
    # Compute the interaction pivot table
    sns.set(font_scale=2)
    fig, ax = plt.subplots()
    pivot_table = pd.pivot_table(
        triplets, values='Score', index='FeatureA', columns='FeatureB', aggfunc=np.mean,
    )
    mask = np.zeros_like(pivot_table.values)
    mask[np.triu_indices_from(mask)] = True
    fsize_heatmap = 20
    if pivot_table.shape[0] > 100:
        sns.set(font_scale=1)
        fsize_heatmap = 3

    logging.info('Visualizing the heatmap ..')

    if pivot_table.shape[0] > 500:
        logging.info(
            'Skipping heatmap visualization due to too many elements ..',
        )
        return

    # Visualize the table
    plt.figure(figsize=(50, 50))
    plt.rcParams.update({'font.size': 1})
    sns.heatmap(
        pivot_table,
        annot=True,
        mask=mask,
        annot_kws={'size': fsize_heatmap},
        square=False,
        cmap='coolwarm',
        linecolor='black',
        linewidths=0.05,
    )
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/heatmap.{image_format}', dpi=500)
    plt.clf()
    plt.cla()
    logging.info(f'Stored heatmap to: {output_folder}/heatmap.{image_format}')


def visualize_barplots(
    triplets: pd.DataFrame,
    output_folder: str,
    reference_json: str,
    image_format: str,
    label: str,
    heuristic: str,
) -> None:
    # Extract only the interactions related to the target attribute
    sns.set(font_scale=8)
    feature_ranks_rows = []
    for enx, row in triplets.iterrows():
        feature_A = row['FeatureA']
        feature_B = row['FeatureB']
        if label in feature_A:
            feature_ranks_rows.append([feature_B, row.Score])
        elif label in feature_B:
            feature_ranks_rows.append([feature_A, row.Score])

    # Align with an existing model
    feature_ranks: pd.DataFrame = pd.DataFrame(feature_ranks_rows)
    feature_ranks.columns = ['Feature', 'Value']
    feature_ranks = feature_ranks[
        ~feature_ranks['Feature'].str.contains(
            label,
        )
    ]
    if not os.path.exists(reference_json):
        reference_json = ''

    if reference_json:
        ref_json = read_reference_json(reference_json)
        used_features = []
        if 'features' in ref_json['desc']:
            for feature in ref_json['desc']['features']:
                used_features.append(feature)

        if 'fields' in ref_json['desc']:
            for field in ref_json['desc']['fields']:
                used_features.append(field)
    else:
        used_features = feature_ranks.keys()

    feature_ranks['Feature'] = feature_ranks['Feature'].astype(str)
    feature_ranks['Value'] = feature_ranks['Value'].astype(float)
    feature_ranks = feature_ranks.groupby(
        by=['Feature'],
    ).median().reset_index()
    feature_ranks = feature_ranks.sort_values(by=['Value'], ascending=False)

    subset_ranges = [10, 25, 50, 100, feature_ranks.shape[0]]
    sns.set_style('whitegrid')

    for subset_range in subset_ranges:
        feature_ranks_reduced = feature_ranks.copy().iloc[:subset_range]
        plt.figure(figsize=(18, 12))
        fig, ax = plt.subplots()

        if (
            feature_ranks_reduced.shape[0] > 45
            and feature_ranks_reduced.shape[0] <= 100
        ):
            ax.yaxis.set_tick_params(labelsize=8)
        elif feature_ranks_reduced.shape[0] > 100:
            ax.yaxis.set_tick_params(labelsize=2)
        else:
            ax.yaxis.set_tick_params(labelsize=25)

        # Visualize the barplot
        plt.title(f'Ranking w.r.t "{label}"\n')
        sns.barplot(
            x='Value',
            y='Feature',
            errwidth=0.7,
            data=feature_ranks_reduced,
            palette='coolwarm_r',
        )

        # Modify the ticks if needed
        for item in ax.get_yticklabels():
            for prod_feature in used_features:
                if item.get_text() in prod_feature:
                    item.set_fontweight('bold')
                    item.set_color('red')
                    break

        plt.xlabel(f'Feature importance (based on heuristic {heuristic})')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(
            f'{output_folder}/barplot_top_{subset_range}.{image_format}', dpi=300,
        )
        plt.clf()
        plt.cla()

        logging.info(
            f'Stored barplot to: {output_folder}/barplot_top_{subset_range}_.{image_format}',
        )


def visualize_all(
    triplets: pd.DataFrame,
    output_folder: str,
    label: str = '',
    reference_json: str = '',
    image_format: str = 'png',
    heuristic: str = 'MI',
) -> None:
    """A method for visualization of the obtained feature interaction maps."""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Visualize feature clusters
    visualize_hierarchical_clusters(triplets, output_folder, image_format)

    # Visualize heatmap
    visualize_heatmap(triplets, output_folder, image_format)

    # visualize barplot
    visualize_barplots(
        triplets, output_folder, reference_json, image_format, label, heuristic,
    )
