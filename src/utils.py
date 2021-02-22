import pandas as pd
import numpy as np

def get_metric_lists():
    cm_metrics_list = ['true_positive_rate_difference','false_positive_rate_difference',
               'false_negative_rate_difference','false_omission_rate_difference',
               'false_discovery_rate_difference','false_positive_rate_ratio',
               'false_negative_rate_ratio','false_omission_rate_ratio',
               'false_discovery_rate_ratio','average_odds_difference',
               'average_abs_odds_difference','error_rate_difference',
               'error_rate_ratio','selection_rate',
               'disparate_impact','statistical_parity_difference',
               'generalized_entropy_index','between_all_groups_generalized_entropy_index',
               'between_group_generalized_entropy_index','theil_index',
               'coefficient_of_variation','between_group_theil_index',
               'between_group_coefficient_of_variation','between_all_groups_theil_index',
               'between_all_groups_coefficient_of_variation','differential_fairness_bias_amplification','accuracy']

    bm_metrics_list = ['consistency','smoothed_empirical_differential_fairness','mean_difference','disparate_impact']

    return cm_metrics_list, bm_metrics_list


def get_metrics_ideal_values():
    cm_zeros = ['true_positive_rate_difference','false_positive_rate_difference',
               'false_negative_rate_difference','false_omission_rate_difference',
               'false_discovery_rate_difference','average_odds_difference',
               'average_abs_odds_difference','error_rate_difference',
               'selection_rate',
               'statistical_parity_difference',
               'generalized_entropy_index','between_all_groups_generalized_entropy_index',
               'between_group_generalized_entropy_index','theil_index',
               'coefficient_of_variation','between_group_theil_index',
               'between_group_coefficient_of_variation','between_all_groups_theil_index',
               'between_all_groups_coefficient_of_variation','differential_fairness_bias_amplification']

    cm_ones = ['false_positive_rate_ratio','false_negative_rate_ratio',
            'false_omission_rate_ratio','false_discovery_rate_ratio',
            'error_rate_ratio','disparate_impact']

    bm_zeros = ['smoothed_empirical_differential_fairness','mean_difference']
    bm_ones = ['disparate_impact','consistency']

    return cm_zeros, cm_ones, bm_zeros, bm_ones