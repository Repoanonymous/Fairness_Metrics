from scipy import stats
import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os
import pickle as pkl
from copy import deepcopy 

def get_sensitivity(datasets, path):
    consistency = {}
    for dataset in datasets:
        results = pkl.load(open(path + dataset + '.pkl', 'rb'))
        results_cm = results['cm']

        if dataset not in consistency.keys():
            consistency[dataset] = {}

        for model in results_cm.keys():
            results_actual = results_cm[model]
            if model not in consistency[dataset].keys():
                consistency[dataset][model] = {}
            for metric in results_actual.keys():
                if metric in ['num_pred_positives','num_pred_negatives', 'accuracy']:
                    continue
                data = results_actual[metric][dataset]
                if metric not in consistency[dataset][model].keys():
                    consistency[dataset][model][metric] = {'median': round(np.nanmedian(data),3), 
                                                            'iqr': round(np.nanquantile(data,0.75)-np.nanquantile(data,0.25),3)}
    return consistency

if __name__ == "__main__":
    path = 'results/'
    datasets = ['Compas','Health','German']
    consistency = get_sensitivity(datasets, path)

    reform = {}
    for key0, value0 in consistency.items():
        for key1,value1 in value0.items():
            for key2,value2 in value1.items():
                for key3,value3 in value2.items():
                    if key2 not in reform.keys():
                        reform[key2] = {}
                    reform[key2][(key0, key1,key3)] = value3

    consistency_df = pd.DataFrame.from_dict(reform, orient = 'index')
    print(consistency_df)