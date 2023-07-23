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
from hyperopt.fmin import generate_trials_to_calculate
import xgboost as xgb
import time
import os
os.chdir('code')
from utilities import *
os.chdir('..')
import sys
import csv
from sdv.sampling import Condition
import warnings
warnings.simplefilter("ignore")


# Defines objective function for the Bayesian optimizer
def objective_maximize(params):
    global clf_auc_history
    global best_val_roc 
    global best_synth
    global first_val_roc
    global first_synth
    global params_range
    N_sim = params["N_sim"]
    if short_epoch:
        params['epochs'] = 1
    # if data_set_name == 'unbalanced_credit_card':
    #     df1 = df_train.loc[df_train[target] == 1]
    #     df0 = df_train.loc[df_train[target] == 0]
    #     synth1 = fit_synth(df_train, params)
    #     synth0 = fit_synth(df_train, params)
    #     synth1.fit(df1)
    #     synth0.fit(df0)
        
    #     num_rows1 = round(N_sim*df[target].mean())
    #     sampled1 = synth1.sample(num_rows = num_rows1)
    #     sampled0 = synth0.sample(num_rows = N_sim - num_rows1)
    #     sampled = pd.concat([sampled1, sampled0], ignore_index=True)
    # else:
    #     synth = fit_synth(df_train, params)
    #     synth.fit(df_train)
    #     sampled = synth.sample(num_rows = N_sim)
    synth = fit_synth(df_train, params)
    synth.fit(df_train)
    sampled = synth.sample(num_rows = N_sim)
    clf_auc = downstream_loss(sampled, df_val, target, classifier = "XGB")
    print(clf_auc)
    
    
    if clf_auc > best_val_roc:
        best_val_roc = clf_auc
        best_synth = sampled
    if clf_auc_history.size == 0:
        output_ = {'val_roc' : [clf_auc]}
        clf_auc_history = pd.DataFrame.from_dict(output_)
        first_synth = sampled
        first_val_roc = best_val_roc
    else:
        output_ = {'val_roc' : [clf_auc]}
        clf_auc_history = pd.concat((clf_auc_history,  pd.DataFrame.from_dict(output_)))

    return {
        'loss' : 1 - clf_auc,
        'status' : STATUS_OK,
        'eval_time ': time.time(),
        'val_roc' : clf_auc,
        }

# The Bayesian optimizer
def trainDT(max_evals:int, method_name):
    global best_val_roc
    global best_synth
    global first_val_roc
    global first_synth
    global clf_auc_history
    global params_range
    if method_name == "GaussianCopula":
        params = get_init_params(method_name)
        N_sim = params["N_sim"]
        synth = fit_synth(df_train, params = params)
        synth.fit(df_train)
        if data_set_name == "unbalanced_credit_card":
            class1 = Condition(num_rows = round(N_sim*df[target].mean()), column_values={target: 1})
            class0 = Condition(num_rows = N_sim - round(N_sim*df[target].mean()), column_values={target: 0})
            best_synth = synth.sample_from_conditions(conditions=[class1,  class0])
        else:
            best_synth = synth.sample(num_rows = N_sim)
        best_val_roc = downstream_loss(best_synth, df_val, target, classifier = "XGB")
        return best_val_roc, best_synth, {}, pd.DataFrame.from_dict({}), best_val_roc, best_synth
    params_range = getparams(method_name)
    clf_auc_history = pd.DataFrame()
    best_val_roc = 0
    init_vals = [get_init_params(method_name)]
    trials = generate_trials_to_calculate(init_vals)
    start = time.time()
    clf_best_param = fmin(fn=objective_maximize,
                    space=params_range,
                    max_evals=max_evals,
                    # rstate=np.random.default_rng(42),
                    early_stop_fn=no_progress_loss(10),
                    algo=tpe.suggest,
                    trials=trials)
    print(clf_best_param)
    print(best_val_roc)
    print('It takes %s minutes' % ((time.time() - start)/60))
    return best_val_roc, best_synth, clf_best_param, clf_auc_history, first_val_roc,first_synth

# Load data and create train test split from the smaller dataset that contains 10% of the full data
arguments = sys.argv
print("arguments: ", arguments)
# assert len(arguments) > 3
if len(arguments) > 2:
    data_set_name = arguments[1]
    target = 'income'
    if data_set_name == 'balanced_credit_card' or data_set_name == 'unbalanced_credit_card':
        target = 'Class'

    method_name = arguments[2]
    encode = eval(arguments[3]) # either categotical or target
    optimization_itr = 350
    short_epoch = False
else:
    data_set_name = 'adult'
    method_name = 'TVAE'
    optimization_itr = 1
    if data_set_name == 'adult':
        target = 'income'
    else:
        target = "Class"
    encode = False
    short_epoch = True


if encode:
    m_name = "encoded_" + data_set_name
else:
    m_name = data_set_name

if os.path.exists("data/output/" + m_name + "_" + method_name + "_clf_best_param_xgboost.csv"):
    raise FileExistsError("This results already exists. Skipping to the next")

df_original = load_data(data_set_name)

df = df_original.copy()
if len(df) > 50000:
    df = df.sample(50000, replace = False, random_state = 5)

df_train, df_val, df_test, encoder = get_train_validation_test_data(df, encode, target)
if not encode:
    df_train.to_csv('data/input/' + m_name + '_train.csv')
    df_val.to_csv('data/input/' + m_name  + '_validation.csv')
    df_test.to_csv('data/input/' + m_name + '_test.csv')

start_time = time.time()
best_val_roc, best_synth, clf_best_param, clf_auc_history, first_val_roc, first_synth = trainDT(max_evals=optimization_itr, method_name=method_name)
elapsed_time = time.time() - start_time

if encode:
    best_synth = encoder.inverse_transform(best_synth)
    best_synth = best_synth[df.columns]
    first_synth = encoder.inverse_transform(first_synth)
    first_synth = first_synth[df.columns]
    df_test = encoder.inverse_transform(df_test)
    df_test = df_test[df.columns]

# Get test auc
test_auc_best = downstream_loss(best_synth, df_test, target, classifier = "XGB")
test_auc_first = downstream_loss(first_synth, df_test, target, classifier = "XGB")

# Save data
clf_best_param["tuned_val_roc"] = best_val_roc
clf_best_param["untuned_val_roc"] = first_val_roc
clf_best_param["tuned_test_roc"] = test_auc_best
clf_best_param["untuned_test_roc"] = test_auc_first
clf_best_param["elapsed_time"] = elapsed_time
df_bparam = pd.DataFrame.from_dict(clf_best_param, orient='index', columns=['Value'])
df_bparam.to_csv("data/output/" + m_name + "_" + method_name + "_clf_best_param_xgboost.csv")
best_synth.to_csv("data/output/" + m_name + "_tuned_" + method_name + "_synthetic_data_xgboost.csv", index = False)
first_synth.to_csv("data/output/" + m_name + "_untuned_" + method_name + "_synthetic_data_xgboost.csv", index = False)
clf_auc_history.to_csv("data/history/" + m_name + "_" + method_name + "_history_auc_score_xgboost.csv")
