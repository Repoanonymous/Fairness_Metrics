from scipy import stats
import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os
import pickle as pkl
from copy import deepcopy 
from scipy import stats

import utils

def load_data(dataset, path):
    results_cm_sorted = {}
    results_dm_sorted = {}
    results_bm_sorted = {}

    
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

    for key in results_cm_base.keys():
        if key not in results_cm_sorted.keys():
            results_cm_sorted[key] = []

        for item in results_cm_base[key][dataset]:
            results_cm_sorted[key].append(item)

    return results_cm_sorted, results_bm_sorted

def lable_fairness(combined_results, _type):
    cm_zeros, cm_ones, bm_zeros, bm_ones = utils.get_metrics_ideal_values()

    print(len(cm_zeros),len(cm_ones))

    if _type == 'cm':
        zeros = cm_zeros
        ones = cm_ones
    elif _type == 'bm':
        zeros = bm_zeros
        ones = bm_ones

    for i in range(combined_results.shape[0]):
        for j in range(combined_results.shape[1]):
            if combined_results.index[i] in zeros:
                if (combined_results.iloc[i,j] >= -0.1) & (combined_results.iloc[i,j] <= 0.1):
                    combined_results.iloc[i,j] = 'Fair'
                else:
                    combined_results.iloc[i,j] = 'Unfair'
            elif combined_results.index[i] in ones:
                if (combined_results.iloc[i,j] >= 0.8) & (combined_results.iloc[i,j] <= 1.2):
                    combined_results.iloc[i,j] = 'Fair'
                else:
                    combined_results.iloc[i,j] = 'Unfair'
    return combined_results
    

if __name__ == "__main__":
    path = '../results'
    combined_results_cm = pd.DataFrame()
    combined_results_bm = pd.DataFrame()
    datasets = ['Adult','Compas','Health','German','bank','Titanic','Student']
    for dataset in datasets:
        results_cm_sorted, results_bm_sorted = load_data(dataset, path)

        results_cm_sorted_df = pd.DataFrame.from_dict(results_cm_sorted, orient = 'index')
        if 'num_pred_positives' in results_cm_sorted_df.index:
            results_cm_sorted_df = results_cm_sorted_df.drop(['num_pred_positives'], axis = 0)
        if 'num_pred_negatives' in results_cm_sorted_df.index:
            results_cm_sorted_df = results_cm_sorted_df.drop(['num_pred_negatives'], axis = 0)
        if 'accuracy' in results_cm_sorted_df.index:
            results_cm_sorted_df = results_cm_sorted_df.drop(['accuracy'], axis = 0)
        results_cm_sorted_df = round(results_cm_sorted_df.median(axis = 1),2)

        results_bm_sorted_df = pd.DataFrame.from_dict(results_bm_sorted, orient = 'index')
        if 'num_pred_positives' in results_bm_sorted_df.index:
            results_bm_sorted_df = results_bm_sorted_df.drop(['num_pred_positives'], axis = 0)
        if 'num_pred_negatives' in results_bm_sorted_df.index:
            results_bm_sorted_df = results_bm_sorted_df.drop(['num_pred_negatives'], axis = 0)
        if 'accuracy' in results_bm_sorted_df.index:
            results_bm_sorted_df = results_bm_sorted_df.drop(['accuracy'], axis = 0)
        results_bm_sorted_df = round(results_bm_sorted_df.median(axis = 1),2)

        combined_results_cm = pd.concat([combined_results_cm,results_cm_sorted_df], axis = 1)
        combined_results_bm = pd.concat([combined_results_bm,results_bm_sorted_df], axis = 1)

    combined_results_cm.columns = datasets
    combined_results_cm = lable_fairness(combined_results_cm, 'cm')

    combined_results_bm.columns = datasets
    combined_results_bm = lable_fairness(combined_results_bm, 'bm')

    print(combined_results_cm)
    print(combined_results_bm)



    