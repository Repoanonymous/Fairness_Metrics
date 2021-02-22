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

    results = pkl.load(open(path + '/' + dataset + '_final_results.pkl', 'rb'))
    
    results_cm_base = results['cm']['baseline']
    results_cm_meta = results['cm']['Meta']
    results_cm_re = results['cm']['Reweighing']

    for method in results['cm'].keys():
        if method not in results_cm_sorted.keys():
            results_cm_sorted[method] = {}
        for key in results['cm'][method].keys():
            if key in ['num_pred_positives','num_pred_negatives','accuracy']:
                continue
            if key not in results_cm_sorted[method].keys():
                results_cm_sorted[method][key] = round(np.nanmedian(results['cm'][method][key][dataset]),3)

    return results_cm_sorted

def measure_fairness(baseline,new):
    cm_zeros, cm_ones, bm_zeros, bm_ones = utils.get_metrics_ideal_values()

    print(len(cm_zeros),len(cm_ones))

    changes = {'FU': 0, 'UF': 0, 'NC': 0}

    datasets = baseline.columns.values.tolist()

    all_changes = {}

    for col in datasets:
        if col not in all_changes.keys():
            all_changes[col] = {'FU': 0, 'UF': 0, 'NC': 0}
    
    for i in range(baseline.shape[0]):
        for j in range(baseline.shape[1]):
            if baseline.index[i] in cm_zeros:
                if abs(baseline.iloc[i,j]) > abs(new.iloc[i,j]):
                    all_changes[datasets[j]]['FU'] += 1
                elif abs(baseline.iloc[i,j]) < abs(new.iloc[i,j]):
                    all_changes[datasets[j]]['UF'] += 1
                else:
                    all_changes[datasets[j]]['NC'] += 1
            elif baseline.index[i] in cm_ones:
                print(i)
    all_changes_df = pd.DataFrame.from_dict(all_changes, orient = 'index')
    print(all_changes_df)
    return all_changes_df
    

if __name__ == "__main__":
    path = '../results'
    baseline = pd.DataFrame()
    meta = pd.DataFrame()
    reweighing = pd.DataFrame()
    datasets = ['Adult','Compas','Health','German','bank','Titanic','Student']
    for dataset in datasets:
        results_cm_sorted = load_data(dataset, path)
        baseline_df = pd.DataFrame.from_dict(results_cm_sorted['baseline'], orient = 'index')
        meta_df = pd.DataFrame.from_dict(results_cm_sorted['Meta'], orient = 'index')
        reweighing_df = pd.DataFrame.from_dict(results_cm_sorted['Reweighing'], orient = 'index')

        baseline = pd.concat([baseline,baseline_df], axis = 1)
        meta = pd.concat([meta,meta_df], axis = 1)
        reweighing = pd.concat([reweighing,reweighing_df], axis = 1)
    
    baseline.columns = datasets
    meta.columns = datasets
    reweighing.columns = datasets

    print(baseline)

    print(meta)


    measure_fairness(baseline,reweighing)
