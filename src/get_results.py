from scipy import stats
import pandas as pd
import random,time,csv
import numpy as np
from numpy import arange
import math,copy,os
import pickle as pkl
from copy import deepcopy 

from multiprocessing import Pool, cpu_count
from threading import Thread
from multiprocessing import Queue


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric,BinaryLabelDatasetMetric,SampleDistortionMetric
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import MetaFairClassifier

import utils

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        #print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def load_data(dataset, label_name, protected_attr):
    path = '../data'
    if dataset == 'Adult':
        dataset_orig = pd.read_csv(path + '/adult.data.csv')

        ## Drop NULL values
        dataset_orig = dataset_orig.dropna()

        ## Drop categorical features
        dataset_orig = dataset_orig.drop(['workclass','fnlwgt','education',
                                          'marital-status','occupation',
                                          'relationship','native-country'],axis=1)

        ## Change symbolics to numerics
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
        dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == ' <=50K', 0, 1)


        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)
        print(dataset_orig.shape)

        # dataset_orig = dataset_orig.drop_duplicates()
        dataset_orig_bld = BinaryLabelDataset(df=dataset_orig, 
                                              label_names=[label_name], 
                                              protected_attribute_names=[protected_attr])
    elif dataset == 'Compas':
        ## Load dataset
        dataset_orig = pd.read_csv(path + '/compas-scores-two-years.csv')

        ## Drop categorical features
        ## Removed two duplicate coumns - 'decile_score','priors_count'
        dataset_orig = dataset_orig.drop(['id','name','first','last','compas_screening_date','dob','age','juv_fel_count','decile_score','juv_misd_count','juv_other_count','days_b_screening_arrest','c_jail_in','c_jail_out','c_case_number','c_offense_date','c_arrest_date','c_days_from_compas','c_charge_desc','is_recid','r_case_number','r_charge_degree','r_days_from_arrest','r_offense_date','r_charge_desc','r_jail_in','r_jail_out','violent_recid','is_violent_recid','vr_case_number','vr_charge_degree','vr_offense_date','vr_charge_desc','type_of_assessment','decile_score','score_text','screening_date','v_type_of_assessment','v_decile_score','v_score_text','v_screening_date','in_custody','out_custody','start','end','event'],axis=1)

        ## Drop NULL values
        dataset_orig = dataset_orig.dropna()


        ## Change symbolics to numerics
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Male', 1, 0)
        dataset_orig['race'] = np.where(dataset_orig['race'] != 'Caucasian', 0, 1)
        dataset_orig['priors_count'] = np.where((dataset_orig['priors_count'] >= 1 ) & (dataset_orig['priors_count'] <= 3), 3, dataset_orig['priors_count'])
        dataset_orig['priors_count'] = np.where(dataset_orig['priors_count'] > 3, 4, dataset_orig['priors_count'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Greater than 45',45,dataset_orig['age_cat'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == '25 - 45', 25, dataset_orig['age_cat'])
        dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Less than 25', 0, dataset_orig['age_cat'])
        dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)

        label_name = 'Probability'
        protected_attr = 'sex'

        ## Rename class column
        dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)

        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)
        
        dataset_orig_bld = BinaryLabelDataset(df=dataset_orig, 
                                              label_names=[label_name], 
                                              protected_attribute_names=[protected_attr])
    elif dataset == 'Health':
        ## Load dataset
        dataset_orig = pd.read_csv(path + '/processed.cleveland.data.csv')

        ## Drop NULL values
        dataset_orig = dataset_orig.dropna()


        ## calculate mean of age column
        mean = dataset_orig.loc[:,"age"].mean()
        dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)

        ## Make goal column binary
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] > 0, 1, 0)


        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)
        
        dataset_orig_bld = BinaryLabelDataset(df=dataset_orig, 
                                              label_names=[label_name], 
                                              protected_attribute_names=[protected_attr])
        
    elif dataset == 'German':
        ## Load dataset
        dataset_orig = pd.read_csv(path + '/GermanData.csv')
        # print(dataset_orig)

        ## Drop categorical features
        dataset_orig = dataset_orig.drop(['1','2','4','5','8','10','11','12','14','15','16','17','18','19','20'],axis=1)

        ## Drop NULL values
        dataset_orig = dataset_orig.dropna()


        ## Change symbolics to numerics
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A91', 1, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A92', 0, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A93', 1, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A94', 1, dataset_orig['sex'])
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A95', 0, dataset_orig['sex'])

        # mean = dataset_orig.loc[:,"age"].mean()
        # dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)
        dataset_orig['age'] = np.where(dataset_orig['age'] >= 25, 1, 0)
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A30', 1, dataset_orig['credit_history'])
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A31', 1, dataset_orig['credit_history'])
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A32', 1, dataset_orig['credit_history'])
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A33', 2, dataset_orig['credit_history'])
        dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A34', 3, dataset_orig['credit_history'])

        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A61', 1, dataset_orig['savings'])
        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A62', 1, dataset_orig['savings'])
        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A63', 2, dataset_orig['savings'])
        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A64', 2, dataset_orig['savings'])
        dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A65', 3, dataset_orig['savings'])

        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A72', 1, dataset_orig['employment'])
        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A73', 1, dataset_orig['employment'])
        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A74', 2, dataset_orig['employment'])
        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A75', 2, dataset_orig['employment'])
        dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A71', 3, dataset_orig['employment'])



        ## ADD Columns
        dataset_orig['credit_history=Delay'] = 0
        dataset_orig['credit_history=None/Paid'] = 0
        dataset_orig['credit_history=Other'] = 0

        dataset_orig['credit_history=Delay'] = np.where(dataset_orig['credit_history'] == 1, 1, dataset_orig['credit_history=Delay'])
        dataset_orig['credit_history=None/Paid'] = np.where(dataset_orig['credit_history'] == 2, 1, dataset_orig['credit_history=None/Paid'])
        dataset_orig['credit_history=Other'] = np.where(dataset_orig['credit_history'] == 3, 1, dataset_orig['credit_history=Other'])

        dataset_orig['savings_more_than_500'] = 0
        dataset_orig['savings_less_than_500'] = 0
        dataset_orig['savings_Unknown'] = 0

        dataset_orig['savings_more_than_500'] = np.where(dataset_orig['savings'] == 1, 1, dataset_orig['savings_more_than_500'])
        dataset_orig['savings_less_than_500'] = np.where(dataset_orig['savings'] == 2, 1, dataset_orig['savings_less_than_500'])
        dataset_orig['savings_Unknown'] = np.where(dataset_orig['savings'] == 3, 1, dataset_orig['savings_Unknown'])

        dataset_orig['employment=1-4 years'] = 0
        dataset_orig['employment=4+ years'] = 0
        dataset_orig['employment=Unemployed'] = 0

        dataset_orig['employment=1-4 years'] = np.where(dataset_orig['employment'] == 1, 1, dataset_orig['employment=1-4 years'])
        dataset_orig['employment=4+ years'] = np.where(dataset_orig['employment'] == 2, 1, dataset_orig['employment=4+ years'])
        dataset_orig['employment=Unemployed'] = np.where(dataset_orig['employment'] == 3, 1, dataset_orig['employment=Unemployed'])


        dataset_orig = dataset_orig.drop(['credit_history','savings','employment'],axis=1)
        ## In dataset 1 means good, 2 means bad for probability. I change 2 to 0
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 2, 0, 1)

        dataset_orig_bld = BinaryLabelDataset(df=dataset_orig, 
                                                  label_names=[label_name], 
                                                  protected_attribute_names=[protected_attr])
        
    elif dataset == 'bank':
        dataset_orig = pd.read_csv(path + '/bank.csv')
        dataset_orig = dataset_orig.drop(['job','marital','education','contact','month','poutcome'],axis=1)

        dataset_orig['default'] = np.where(dataset_orig['default'] == 'no', 0, 1)
        dataset_orig['housing'] = np.where(dataset_orig['housing'] == 'no', 0, 1)
        dataset_orig['loan'] = np.where(dataset_orig['loan'] == 'no', 0, 1)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 'yes', 1, 0)

        mean = dataset_orig.loc[:,"age"].mean()
        dataset_orig['age'] = np.where(dataset_orig['age'] >= 30, 1, 0)
        print(mean)


        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)
    
        dataset_orig_bld = BinaryLabelDataset(df=dataset_orig, 
                                                  label_names=[label_name], 
                                                  protected_attribute_names=[protected_attr])
        
    elif dataset == 'default':
        dataset_orig = pd.read_csv(path + '/default_of_credit_card_clients_first_row_removed.csv')

        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 2, 0,1)


        ## Drop NULL values
        dataset_orig = dataset_orig.dropna()

        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)
        
        dataset_orig_bld = BinaryLabelDataset(df=dataset_orig, 
                                                  label_names=[label_name], 
                                                  protected_attribute_names=[protected_attr])
        
    elif dataset == 'credit':
        eta = 0.2
        dataset_orig = pd.read_csv(path + '/Home Credit Default Risk.csv')

        dataset_orig = dataset_orig.drop(['SK_ID_CURR','NAME_CONTRACT_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY',
                                          'NAME_TYPE_SUITE','NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE' ,
                                          'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE','OCCUPATION_TYPE',
                                          'WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','FONDKAPREMONT_MODE',
                                          'HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE',
                                          'DAYS_LAST_PHONE_CHANGE'],axis=1)
        dataset_orig = dataset_orig[0:30751]

        dataset_orig = dataset_orig.fillna(0)
        dataset_orig['CODE_GENDER'] = np.where(dataset_orig['CODE_GENDER'] == 'M', 1, 0)
        
        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)
        dataset_orig_bld = BinaryLabelDataset(df=dataset_orig, 
                                                  label_names=[label_name], 
                                                  protected_attribute_names=[protected_attr])
        
    elif dataset == 'MEPS15':
        dataset_orig = pd.read_csv(path + '/MEPS/h181.csv')

        dataset_orig = dataset_orig.dropna()

        dataset_orig = dataset_orig.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                                      'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                                      'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                                      'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                                      'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})


        dataset_orig = dataset_orig[dataset_orig['PANEL'] == 20]
        dataset_orig = dataset_orig[dataset_orig['REGION'] >= 0] # remove values -1
        dataset_orig = dataset_orig[dataset_orig['AGE'] >= 0] # remove values -1
        dataset_orig = dataset_orig[dataset_orig['MARRY'] >= 0] # remove values -1, -7, -8, -9
        dataset_orig = dataset_orig[dataset_orig['ASTHDX'] >= 0] # remove values -1, -7, -8, -9
        dataset_orig = dataset_orig[(dataset_orig[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                                     'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                     'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                     'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                                     'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]

        dataset_orig['RACEV2X'] = np.where((dataset_orig['HISPANX'] == 2 ) & (dataset_orig['RACEV2X'] == 1), 1, dataset_orig['RACEV2X'])
        dataset_orig['RACEV2X'] = np.where(dataset_orig['RACEV2X'] != 1 , 0, dataset_orig['RACEV2X'])
        dataset_orig = dataset_orig.rename(columns={"RACEV2X" : "RACE"})
        # dataset_orig['UTILIZATION'] = np.where(dataset_orig['UTILIZATION'] >= 10, 1, 0)



        def utilization(row):
                return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

        dataset_orig['TOTEXP15'] = dataset_orig.apply(lambda row: utilization(row), axis=1)
        lessE = dataset_orig['TOTEXP15'] < 10.0
        dataset_orig.loc[lessE,'TOTEXP15'] = 0.0
        moreE = dataset_orig['TOTEXP15'] >= 10.0
        dataset_orig.loc[moreE,'TOTEXP15'] = 1.0

        dataset_orig = dataset_orig.rename(columns = {'TOTEXP15' : 'UTILIZATION'})

        dataset_orig = dataset_orig[['REGION','AGE','SEX','RACE','MARRY',
                                         'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                         'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                         'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                         'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42',
                                         'PCS42','MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION', 'PERWT15F']]

        dataset_orig = dataset_orig.rename(columns={"UTILIZATION": "Probability","RACE" : "race"})
        

        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)
        dataset_orig_bld = BinaryLabelDataset(df=dataset_orig, 
                                                  label_names=[label_name], 
                                                  protected_attribute_names=[protected_attr])
        
    elif dataset == 'MEPS16':
        dataset_orig = pd.read_csv(path + '/MEPS/h192.csv')

        # ## Drop NULL values
        dataset_orig = dataset_orig.dropna()


        dataset_orig = dataset_orig.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                                      'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                                      'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                                      'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                                      'POVCAT16' : 'POVCAT', 'INSCOV16' : 'INSCOV'})


        dataset_orig = dataset_orig[dataset_orig['PANEL'] == 21]
        dataset_orig = dataset_orig[dataset_orig['REGION'] >= 0] # remove values -1
        dataset_orig = dataset_orig[dataset_orig['AGE'] >= 0] # remove values -1
        dataset_orig = dataset_orig[dataset_orig['MARRY'] >= 0] # remove values -1, -7, -8, -9
        dataset_orig = dataset_orig[dataset_orig['ASTHDX'] >= 0] # remove values -1, -7, -8, -9
        dataset_orig = dataset_orig[(dataset_orig[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                                 'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]

        # ## Change symbolics to numerics
        dataset_orig['RACEV2X'] = np.where((dataset_orig['HISPANX'] == 2 ) & (dataset_orig['RACEV2X'] == 1), 1, dataset_orig['RACEV2X'])
        dataset_orig['RACEV2X'] = np.where(dataset_orig['RACEV2X'] != 1 , 0, dataset_orig['RACEV2X'])
        dataset_orig = dataset_orig.rename(columns={"RACEV2X" : "RACE"})
        # dataset_orig['UTILIZATION'] = np.where(dataset_orig['UTILIZATION'] >= 10, 1, 0)



        def utilization(row):
                return row['OBTOTV16'] + row['OPTOTV16'] + row['ERTOT16'] + row['IPNGTD16'] + row['HHTOTD16']

        dataset_orig['TOTEXP16'] = dataset_orig.apply(lambda row: utilization(row), axis=1)
        lessE = dataset_orig['TOTEXP16'] < 10.0
        dataset_orig.loc[lessE,'TOTEXP16'] = 0.0
        moreE = dataset_orig['TOTEXP16'] >= 10.0
        dataset_orig.loc[moreE,'TOTEXP16'] = 1.0

        dataset_orig = dataset_orig.rename(columns = {'TOTEXP16' : 'UTILIZATION'})

        dataset_orig = dataset_orig[['REGION','AGE','SEX','RACE','MARRY',
                                         'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                         'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                         'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                         'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42',
                                         'PCS42','MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION', 'PERWT16F']]

        dataset_orig = dataset_orig.rename(columns={"UTILIZATION": "Probability","RACE" : "race"})

        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)
        dataset_orig_bld = BinaryLabelDataset(df=dataset_orig, 
                                                  label_names=[label_name], 
                                                  protected_attribute_names=[protected_attr])
    elif dataset == 'Student':
        dataset_orig = pd.read_csv(path + '/Student.csv')

        ## Drop NULL values
        dataset_orig = dataset_orig.dropna()

        ## Drop categorical features
        dataset_orig = dataset_orig.drop(['school','address', 'famsize', 'Pstatus','Mjob', 'Fjob', 'reason', 'guardian'],axis=1)

        ## Change symbolics to numerics
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'M', 1, 0)
        dataset_orig['schoolsup'] = np.where(dataset_orig['schoolsup'] == 'yes', 1, 0)
        dataset_orig['famsup'] = np.where(dataset_orig['famsup'] == 'yes', 1, 0)
        dataset_orig['paid'] = np.where(dataset_orig['paid'] == 'yes', 1, 0)
        dataset_orig['activities'] = np.where(dataset_orig['activities'] == 'yes', 1, 0)
        dataset_orig['nursery'] = np.where(dataset_orig['nursery'] == 'yes', 1, 0)
        dataset_orig['higher'] = np.where(dataset_orig['higher'] == 'yes', 1, 0)
        dataset_orig['internet'] = np.where(dataset_orig['internet'] == 'yes', 1, 0)
        dataset_orig['romantic'] = np.where(dataset_orig['romantic'] == 'yes', 1, 0)
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] > 12, 1, 0)

        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)
        dataset_orig_bld = BinaryLabelDataset(df=dataset_orig, 
                                                  label_names=[label_name], 
                                                  protected_attribute_names=[protected_attr])
        
    elif dataset == 'Titanic':
        dataset_orig = pd.read_csv(path + '/Titanic.csv')

        ## Drop NULL values
        dataset_orig = dataset_orig.fillna(0)

        ## Drop categorical features
        dataset_orig = dataset_orig.drop(['Name','Ticket','Cabin','PassengerId','Embarked'],axis=1)

        ## Change symbolics to numerics
        dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'male', 1, 0)

        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)
        dataset_orig_bld = BinaryLabelDataset(df=dataset_orig, 
                                                  label_names=[label_name], 
                                                  protected_attribute_names=[protected_attr])


        
    return dataset_orig, dataset_orig_bld
        

def meta_model(train_bld, test_bld, val_bld, protected_attr, label_name):
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    
    tau_values = {}
    for i in arange(0,1,0.2):
        try:
            tau = i
            debiased_model = MetaFairClassifier(tau=tau, sensitive_attr=protected_attr,type="sr")
            debiased_model.fit(train_bld)

            actual_test_bld_pred = debiased_model.predict(val_bld)

            cm = ClassificationMetric(val_bld, actual_test_bld_pred,
                                  unprivileged_groups=[{protected_attr: 0}],
                                  privileged_groups=[{protected_attr: 1}])
            tau_values[cm.average_odds_difference()] = tau
        except IndexError as e:
            continue
          
    tau = tau_values[min(tau_values)]
    
    debiased_model = MetaFairClassifier(tau=tau, sensitive_attr=protected_attr,type="sr")
    debiased_model.fit(train_bld)
    
    actual_test_bld_pred = debiased_model.predict(test_bld)
    
    cm = ClassificationMetric(test_bld, actual_test_bld_pred,
                          unprivileged_groups=[{protected_attr: 0}],
                          privileged_groups=[{protected_attr: 1}])

    
    return cm


def Reweighing_model(X_train,X_test,y_train,y_test, protected_attr, label_name):
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    columns = X_train.columns
    
    actual_train = copy.deepcopy(X_train)
    actual_train[label_name] = y_train
    actual_train_bld = BinaryLabelDataset(df=actual_train, 
                                         label_names=[label_name], 
                                         protected_attribute_names=[protected_attr])
                                          
    actual_test = copy.deepcopy(X_test)
    actual_test[label_name] = y_test
    actual_test_bld = BinaryLabelDataset(df=actual_test, 
                                     label_names=[label_name], 
                                     protected_attribute_names=[protected_attr])
    
    clf_w = Reweighing(unprivileged_groups=[{protected_attr: 0}],
                              privileged_groups=[{protected_attr: 1}])
    
    clf_w.fit(actual_train_bld)
    temp_bld = clf_w.transform(actual_train_bld)
    
    lmod = LogisticRegression(max_iter=1000)
    lmod.fit(X_train, y_train, 
             sample_weight=temp_bld.instance_weights)
    y_test_pred = lmod.predict(X_test)

    actual_test_bld_pred = actual_test_bld.copy(deepcopy=True)
    actual_test_bld_pred.labels = y_test_pred
 
    actual_train = copy.deepcopy(X_train)
    actual_train[label_name] = y_train
    actual_train_bld = BinaryLabelDataset(df=actual_train, 
                                         label_names=[label_name], 
                                         protected_attribute_names=[protected_attr])
        
    cm = ClassificationMetric(actual_test_bld, actual_test_bld_pred,
                          unprivileged_groups=[{protected_attr: 0}],
                          privileged_groups=[{protected_attr: 1}])

    bm = BinaryLabelDatasetMetric(temp_bld,unprivileged_groups=[{protected_attr: 0}],
                          privileged_groups=[{protected_attr: 1}])
    
    return cm, bm


def baseline_model(X_train,X_test,y_train,y_test, protected_attr, label_name):
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    
    actual_train = copy.deepcopy(X_train)
    actual_train[label_name] = y_train
    actual_train_bld = BinaryLabelDataset(df=actual_train, 
                                         label_names=[label_name], 
                                         protected_attribute_names=[protected_attr])
    
    actual_test = copy.deepcopy(X_test)
    actual_test[label_name] = y_test
    actual_test_bld = BinaryLabelDataset(df=actual_test, 
                                     label_names=[label_name], 
                                     protected_attribute_names=[protected_attr])
    
    lmod = LogisticRegression(max_iter=1000)
    lmod.fit(X_train, y_train)
    y_test_pred = lmod.predict(X_test)

    actual_test_bld_pred = actual_test_bld.copy()
    actual_test_bld_pred.labels = y_test_pred
    
    cm = ClassificationMetric(actual_test_bld, actual_test_bld_pred,
                          unprivileged_groups=[{protected_attr: 0}],
                          privileged_groups=[{protected_attr: 1}])

    bm = BinaryLabelDatasetMetric(actual_train_bld,unprivileged_groups=[{protected_attr: 0}],
                          privileged_groups=[{protected_attr: 1}])
    return cm, bm

def run(dataset, protected_attr, label_name, cm_metrics_list, bm_metrics_list):
    path = '../results/'
    results_cm_basaeline = {}
    results_cm_PrejudiceRemover = {}
    results_cm_Reweighing = {}
    results_cm_MetaClassifier = {}
    results_bm_basaeline = {}
    results_bm_Reweighing = {}
    results_dm = {}
    dataset_orig,dataset_orig_bld = load_data(dataset, label_name, protected_attr)
    for j in range(5):
        print("+++++++++++++++++++++++",j)
        kf = KFold(n_splits=5)
        y = dataset_orig.Probability
        X = dataset_orig.drop([label_name], axis = 1)
        for train_index, test_index in kf.split(X):
            try:
                X_train, X_test = X.loc[train_index], X.loc[test_index]
                y_train, y_test = y.loc[train_index], y.loc[test_index]
                
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
                
                train_bld = copy.deepcopy(X_train)
                train_bld[label_name] = y_train
                
                dataset_train_bld = BinaryLabelDataset(df=train_bld, 
                                      label_names=[label_name], 
                                      protected_attribute_names=[protected_attr])
                
                test_bld = copy.deepcopy(X_test)
                test_bld[label_name] = y_test
                
                dataset_test_bld = BinaryLabelDataset(df=test_bld, 
                                      label_names=[label_name], 
                                      protected_attribute_names=[protected_attr])
                
                val_bld = copy.deepcopy(X_val)
                val_bld[label_name] = y_val
                
                dataset_val_bld = BinaryLabelDataset(df=val_bld, 
                                      label_names=[label_name], 
                                      protected_attribute_names=[protected_attr])
                

                cm, bm = baseline_model(X_train,X_test,y_train,y_test, 
                                        protected_attr, label_name)


                
                    
                cm_fair_meta = meta_model(dataset_train_bld, dataset_test_bld,dataset_val_bld,
                                                   protected_attr, label_name)

                

                cm_fair_2, bm_fair_2 = Reweighing_model(X_train,X_test,y_train,y_test, 
                                                                   protected_attr, label_name)
                for metric in cm_metrics_list:
                    if metric not in results_cm_MetaClassifier.keys():
                        results_cm_MetaClassifier[metric] = {}
                    if dataset not in results_cm_MetaClassifier[metric].keys():
                        results_cm_MetaClassifier[metric][dataset] = []
                    method_to_call = getattr(cm_fair_meta, metric)
                    res = method_to_call()
                    results_cm_MetaClassifier[metric][dataset].append(res)
                
                
                
                for metric in cm_metrics_list:
                    if metric not in results_cm_basaeline.keys():
                        results_cm_basaeline[metric] = {}
                    if dataset not in results_cm_basaeline[metric].keys():
                        results_cm_basaeline[metric][dataset] = []
                    method_to_call = getattr(cm, metric)
                    results_cm_basaeline[metric][dataset].append(method_to_call())

                for metric in bm_metrics_list:
                    if metric not in results_bm_basaeline.keys():
                        results_bm_basaeline[metric] = {}
                    if dataset not in results_bm_basaeline[metric].keys():
                        results_bm_basaeline[metric][dataset] = []
                    method_to_call = getattr(bm, metric)
                    results_bm_basaeline[metric][dataset].append(method_to_call())


                for metric in cm_metrics_list:
                    if metric not in results_cm_Reweighing.keys():
                        results_cm_Reweighing[metric] = {}
                    if dataset not in results_cm_Reweighing[metric].keys():
                        results_cm_Reweighing[metric][dataset] = []
                    method_to_call = getattr(cm_fair_2, metric)
                    results_cm_Reweighing[metric][dataset].append(method_to_call())

                for metric in bm_metrics_list:
                    if metric not in results_bm_Reweighing.keys():
                        results_bm_Reweighing[metric] = {}
                    if dataset not in results_bm_Reweighing[metric].keys():
                        results_bm_Reweighing[metric][dataset] = []
                    method_to_call = getattr(bm_fair_2, metric)
                    results_bm_Reweighing[metric][dataset].append(method_to_call())
            except Exception as e:
                print(e,dataset)
                continue
    results_cm = {'baseline':results_cm_basaeline,
                  'Reweighing':results_cm_Reweighing,
                  'Meta': results_cm_MetaClassifier}
    results_bm = {'baseline':results_bm_basaeline,
                  'Reweighing':results_bm_Reweighing}
    results = {'cm':results_cm,'bm':results_bm}
    with open(path + dataset + '.pkl', 'wb') as handle:
        pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    label_name = 'Probability'
    datasets = ['Adult','Compas','Health','German','bank','Student','Titanic']
    protected_attrs = ['sex','sex','sex','sex','age','sex','sex']
    threads = []
    cm_metrics_list,bm_metrics_list = utils.get_metric_lists()
    
    for i in range(len(datasets)):
        dataset = datasets[i]
        protected_attr = protected_attrs[i]
        print("starting thread ",i)
        t = ThreadWithReturnValue(target = run, args = [dataset,protected_attr,label_name,cm_metrics_list,bm_metrics_list])
        threads.append(t)
    for th in threads:
        th.start()
    for th in threads:
        response = th.join()