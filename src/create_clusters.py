from scipy import stats
import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os
import pickle as pkl
from copy import deepcopy 
from scipy import stats

from sklearn.cluster import AgglomerativeClustering

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data(datasets, path):
    results_cm_sorted = {}
    results_dm_sorted = {}
    results_bm_sorted = {}

    for dataset in datasets:
        results = pkl.load(open(path + '/' + dataset + '_final_results.pkl', 'rb'))
        
        results_cm_base = results['cm']['baseline']
        results_cm_meta = results['cm']['Meta']
        results_cm_re = results['cm']['Reweighing']

        results_bm_baseline = results['bm']['baseline']
        results_bm_re = results['bm']['Reweighing']

        for key in results_bm_baseline.keys():
            if key not in results_bm_sorted.keys():
                results_bm_sorted[key] = []

            for item in results_bm_baseline[key][dataset]:
                if key == 'consistency':
                    item = item[0]
                results_bm_sorted[key].append(item)
            
        for key in results_bm_re.keys():
            if key not in results_bm_sorted.keys():
                results_bm_sorted[key] = []

            for item in results_bm_re[key][dataset]:
                if key == 'consistency':
                    item = item[0]
                results_bm_sorted[key].append(item)

        for key in results_cm_base.keys():
            if key not in results_cm_sorted.keys():
                results_cm_sorted[key] = []

            for item in results_cm_base[key][dataset]:
                results_cm_sorted[key].append(item)

        for key in results_cm_meta.keys():
            if key not in results_cm_sorted.keys():
                results_cm_sorted[key] = []

            for item in results_cm_meta[key][dataset]:
                results_cm_sorted[key].append(item)

        for key in results_cm_re.keys():
            if key not in results_cm_sorted.keys():
                results_cm_sorted[key] = []

            for item in results_cm_re[key][dataset]:
                results_cm_sorted[key].append(item)

    return results_cm_sorted, results_bm_sorted

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    dendrogram(linkage_matrix, ax=ax, **kwargs)
    plt.xlabel('Clusters',size=16)
    plt.ylabel('Dissimilarity',size=16)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='y', which='major', labelsize=16)


def create_cm_clusters(results_cm_sorted):
    results_df_cm = pd.DataFrame.from_dict(results_cm_sorted, orient = 'index')
    if 'num_pred_positives' in results_df_cm.index:
        results_df_cm = results_df_cm.drop(['num_pred_positives'], axis = 0)
    if 'num_pred_negatives' in results_df_cm.index:
        results_df_cm = results_df_cm.drop(['num_pred_negatives'], axis = 0)
    if 'accuracy' in results_df_cm.index:
        results_df_cm = results_df_cm.drop(['accuracy'], axis = 0)
    results_df_cm = results_df_cm.T
    results_df_cm = results_df_cm.dropna()
    results_df_cm_corr = results_df_cm.corr('spearman')

    results_df_cm_corr = results_df_cm_corr.abs()
    results_df_cm_corr = 1-results_df_cm_corr

    cols = results_df_cm_corr.columns

    clustering_cm = AgglomerativeClustering(n_clusters = None, 
                                        distance_threshold = 0.6,
                                        affinity = 'precomputed',
                                        linkage='average').fit(results_df_cm_corr)

    clusters = zip(results_df_cm_corr.index,clustering_cm.labels_)
    clusters_df = pd.DataFrame(clusters, columns = ['name','clusters'])
    clusters_df = clusters_df.sort_values('clusters')
    print(clusters_df.sort_values('clusters'))
    plot_dendrogram(clustering_cm)


def create_bm_clusters(results_bm_sorted):
    results_df_bm = pd.DataFrame.from_dict(results_bm_sorted, orient = 'index')
    results_df_bm = results_df_bm.T
    results_df_bm = results_df_bm.dropna()
    results_df_bm_corr = results_df_bm.corr('spearman')

    results_df_bm_corr = results_df_bm_corr.abs()
    results_df_bm_corr = 1-results_df_bm_corr

    clustering_bm = AgglomerativeClustering(n_clusters = None, 
                                        distance_threshold = 0.7,
                                        affinity = 'precomputed',
                                        linkage='average').fit(results_df_bm_corr)
    clusters = zip(results_df_bm_corr.index,clustering_bm.labels_)
    clusters_df_bm = pd.DataFrame(clusters, columns = ['name','clusters'])
    clusters_df_bm = clusters_df_bm.sort_values('clusters')
    print(clusters_df_bm)
    plot_dendrogram(clustering_bm)


if __name__ == "__main__":
    path = '../results'
    datasets = ['Adult','Compas','Health','German','bank','Titanic','Student']
    results_cm_sorted, results_bm_sorted = load_data(datasets, path)
    create_cm_clusters(results_cm_sorted)
    create_bm_clusters(results_bm_sorted)