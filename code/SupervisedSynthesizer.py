import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.tree import DecisionTreeClassifier
from hyperopt.early_stop import no_progress_loss
import xgboost as xgb
import time
import os
os.chdir('code')
from utilities import *
os.chdir('..')
import sys
import csv

import warnings
warnings.simplefilter("ignore")


# Defines objective function for the Bayesian optimizer
def objective_maximize(params):
    global clf_auc_history
    global best_test_roc 
    global best_synth
    global params_range
    synth = fit_synth(df_train, params)
    synth.fit(df_train)

    N_sim = params["N_sim"]
    sampled = synth.sample(num_rows = N_sim)
    clf_auc = downstream_loss(sampled, df_test, target, classifier = "XGB")
    print(clf_auc)
    if clf_auc > best_test_roc:
        best_test_roc = clf_auc
        best_synth = sampled
    if clf_auc_history.size == 0:
        output_ = {'test_roc' : [clf_auc]}
        clf_auc_history = pd.DataFrame.from_dict(output_)
    else:
        output_ = {'test_roc' : [clf_auc]}
        clf_auc_history = pd.concat((clf_auc_history,  pd.DataFrame.from_dict(output_)))

    return {
        'loss' : 1 - clf_auc,
        'status' : STATUS_OK,
        'eval_time ': time.time(),
        'test_roc' : clf_auc,
        }

# The Bayesian optimizer
def trainDT(max_evals:int, method_name):
    global best_test_roc
    global best_synth
    global clf_auc_history
    global params_range
    params_range = getparams(method_name)
    clf_auc_history = pd.DataFrame()
    best_test_roc = 0
    trials = Trials()
    start = time.time()
    clf_best_param = fmin(fn=objective_maximize,
                    space=params_range,
                    max_evals=max_evals,
                    # rstate=np.random.default_rng(42),
                    early_stop_fn=no_progress_loss(10),
                    algo=tpe.suggest,
                    trials=trials)
    print(clf_best_param)
    print(best_test_roc)
    print('It takes %s minutes' % ((time.time() - start)/60))
    return best_test_roc, best_synth, clf_best_param, clf_auc_history

# Load data and create train test split from the smaller dataset that contains 10% of the full data
arguments = sys.argv
print("arguments: ", arguments)
# assert len(arguments) > 3
if len(arguments) > 3:
    data_set_name = arguments[1]
    target = 'income'
    if data_set_name == 'balanced_credit_card' or data_set_name == 'unbalanced_credit_card':
        target = 'Class'

    method_name = arguments[2]
    optimization_itr = int(arguments[3])
else:
    data_set_name = 'adult'
    method_name = 'TVAE'
    optimization_itr = 350
    target = 'income'

df_original = load_data(data_set_name)

df = df_original.copy()
if len(df) > 50000:
    df = df.sample(50000, replace = False, random_state = 5)

df_train, df_test = train_test_split(df, test_size = 0.3,  random_state = 5) #70% is training and 30 to test
df_test, df_val = train_test_split(df_test, test_size = 1 - 0.666,  random_state = 5)# out of 30, 20 is test and 10 for validation



# df = df_original.copy()
# df_modified = target_encoder.transform(df)

# for col in cat_col:
#     df[col] = df[col].astype('category')
# df, df_te = train_test_split(df, test_size = 0.2,  random_state = 5)
# df.to_csv("../data/train.csv", index=False)
# df_te.to_csv("../data/test.csv", index=False)
# target = 'income'

# x_train = df.loc[:, df.columns != target]
# y_train = df[target]

# x_test = df_te.loc[:, df_te.columns != target]
# y_test = df_te[target]


params_range = getparams(method_name)
start_time = time.time()
best_test_roc, best_synth, clf_best_param, clf_auc_history = trainDT(max_evals=optimization_itr, method_name=method_name)
elapsed_time = time.time() - start_time
best_test_roc

best_synth

clf_auc_history

validation_auc = downstream_loss(best_synth, df_val, target, classifier = "XGB")

clf_best_param["test_roc"] = best_test_roc
index = [0]  # Modify this based on your requirements
df_bparam = pd.DataFrame.from_dict(clf_best_param, orient='index', columns=['Value'])
df_bparam.to_csv("data/output/" + data_set_name + "_tuned_" + method_name + "_clf_best_param_xgboost.csv", index=False)
best_synth.to_csv("data/output/" + data_set_name + "_tuned_" + method_name + "_synthetic_data_xgboost.csv", index=False)
clf_auc_history.to_csv("data/history/" + data_set_name + "_tuned_" + method_name + "_history_auc_score_xgboost.csv", index=False)

# Create a DataFrame with a single row and column
df = pd.DataFrame([validation_auc, elapsed_time], columns=['values'], index = ['valudation_auc', "elapsed_time"])
df.to_csv("data/output/" + data_set_name + "_tuned_" + method_name + 'validation_auc.csv', index=False)