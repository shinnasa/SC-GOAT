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
# Ger user defined auguments
##################################################################################################
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
otupath = "data/output/ES10/"

##################################################################################################
# Load data
##################################################################################################
df_original = load_data(data_set_name)

df = df_original.copy()
if len(df) > 50000:
    df = df.sample(50000, replace = False, random_state = 5)

df_train, df_val, df_test, encoder = get_train_validation_test_data(df, encode, target)

##################################################################################################
# Ger tuned and untuned hp
###################################################
################################################
# def get_tuned_params(method_name, outpath):
outpath = "data/output/"
data_set_name = 'adult'
method_name = 'CTGAN'

get_tuned_params(method_name, outpath)
#return params

init_params = get_init_params(method_name)
init_params['epochs'] = 10
synth = fit_synth(df_train, init_params)
synth.fit(df_train)

##################################################################################################
# Perform simulation
##################################################################################################

lauc = []
N_sim = 10000
for i in range(50):
    sampled = synth.sample(num_rows = N_sim)
    if encode:
        df_test = encoder.inverse_transform(df_test)
        df_test = df_test[df.columns]
    clf_auc = downstream_loss(sampled, df_test, target, classifier = "XGB")
    lauc.append(clf_auc)




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
