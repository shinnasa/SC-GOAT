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

##################################################################################################
# Define arguments for the python script
##################################################################################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--DATA_SET_NAME", help="DATA_SET_NAME can be ['adult', 'unbalanced_credit_card', 'balanced_credit_card']",  default='adult', choices=["adult", "unbalanced_credit_card", "balanced_credit_card"])
parser.add_argument("-m", "--METHOD_NAME", help="METHOD_NAME can be ['CopulaGAN', 'CTGAN', 'GaussianCopula', 'TVAE']", default="GaussianCopula", choices = ['CopulaGAN', 'CTGAN', 'GaussianCopula', 'TVAE'])
parser.add_argument("-e", "--ENCODE", help="ENCODE can be [True, False]", default="False", choices = ["True", "False"])
parser.add_argument("-i", "--ITR", help="Number of Optimization Iterations", type=int, default=350)
parser.add_argument("-o", "--OUTPUT_DIR", help="Output directory", required=True)
args = parser.parse_args()

##################################################################################################
# Define functions for Bayesian Optimization
##################################################################################################
def objective_maximize(params):
    global clf_auc_history
    global best_val_roc 
    global best_synth
    global first_val_roc
    global first_synth
    global params_range
    global best_hp
    N_sim = params["N_sim"]
    if short_epoch:
        params['epochs'] = 1
    synth = fit_synth(df_train, params)
    synth.fit(df_train)
    sampled = synth.sample(num_rows = N_sim)
    clf_auc = downstream_loss(sampled, df_val, target, classifier = "XGB")
    print(clf_auc)
    
    if clf_auc > best_val_roc:
        best_val_roc = clf_auc
        best_synth = sampled
        best_hp = params
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
    global best_hp
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
        return best_val_roc, best_synth, {}, pd.DataFrame.from_dict({}), best_val_roc, best_synth, {}
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
    return best_val_roc, best_synth, clf_best_param, clf_auc_history, first_val_roc, first_synth, best_hp

##################################################################################################
# Get user defined arguments
##################################################################################################
print("arguments: ", args)
data_set_name = args.DATA_SET_NAME
method_name = args.METHOD_NAME
encode = eval(args.ENCODE)
optimization_itr = args.ITR
short_epoch = False
output_dir = args.OUTPUT_DIR

target = 'income'
if data_set_name == 'balanced_credit_card' or data_set_name == 'unbalanced_credit_card':
    target = 'Class'

print("data_set_name: ", data_set_name)
print("target: ", target)
print("method_name: ", method_name)
print("encode: ", encode)
print("optimization_itr: ", optimization_itr)
print("output_dir: ", output_dir)

if encode:
    m_name = "encoded_" + data_set_name
else:
    m_name = data_set_name

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.mkdir(output_dir + "/output/")
    os.mkdir(output_dir + "/history/")

if not os.path.exists(output_dir + "/output/"):
    os.mkdir(output_dir + "/output/")

if not os.path.exists(output_dir + "/history/"):
    os.mkdir(output_dir + "/history/")

if os.path.exists(output_dir + "/output/" + m_name + "_" + method_name + "_clf_best_param_xgboost.csv"):
    raise FileExistsError("This results already exists. Skipping to the next")

##################################################################################################
# Load data
##################################################################################################
df_original = load_data(data_set_name)

df = df_original.copy()
if len(df) > 50000:
    df = df.sample(50000, replace = False, random_state = 5)

df_train, df_val, df_test, encoder = get_train_validation_test_data(df, encode, target)
if not encode:
    df_train.to_csv('data/input/' + m_name + '_train.csv')
    df_val.to_csv('data/input/' + m_name  + '_validation.csv')
    df_test.to_csv('data/input/' + m_name + '_test.csv')

##################################################################################################
# Run Bayesian Optimization
##################################################################################################
start_time = time.time()
best_val_roc, best_synth, clf_best_param, clf_auc_history, first_val_roc, first_synth, best_hp = trainDT(max_evals=optimization_itr, method_name=method_name)
print("best_hp: ", best_hp)
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

##################################################################################################
# Save Results
##################################################################################################
clf_best_param["tuned_val_roc"] = best_val_roc[1]
clf_best_param["untuned_val_roc"] = first_val_roc[1]
clf_best_param["tuned_test_roc"] = test_auc_best[1]
clf_best_param["untuned_test_roc"] = test_auc_first[1]
clf_best_param["elapsed_time"] = elapsed_time
print("clf_best_param: ", clf_best_param)
df_bparam = pd.DataFrame.from_dict(clf_best_param, orient='index', columns=['Value'])
df_bparam.to_csv(output_dir + "/output/" + m_name + "_" + method_name + "_clf_best_param_xgboost.csv")

df_bhp = pd.DataFrame.from_dict(clf_best_param, orient='index', columns=['Value'])
df_bhp.to_csv(output_dir + "/output/" + m_name + "_" + method_name + "_clf_best_hp_xgboost.csv")

best_synth.to_csv(output_dir + "/output/" + m_name + "_tuned_" + method_name + "_synthetic_data_xgboost.csv", index = False)
first_synth.to_csv(output_dir + "/output/" + m_name + "_untuned_" + method_name + "_synthetic_data_xgboost.csv", index = False)
clf_auc_history.to_csv(output_dir + "/history/" + m_name + "_" + method_name + "_history_auc_score_xgboost.csv")
