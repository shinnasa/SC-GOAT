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
import xgboost as xgb
import time
import os
os.chdir('code')
from utilities import *
os.chdir('..')
import sys

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
    method_name = 'CTGAN'
    optimization_itr = 1000

df_original, target = load_data(data_set_name)

df = df_original.copy()

df_train, df_test = train_test_split(df, test_size = 0.2,  random_state = 5)

x_train = df_train.loc[:, df_train.columns != target]
y_train = df_train[target]

x_test = df_test.loc[:, df_test.columns != target]
y_test = df_test[target]

x_train

x_test

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
best_test_roc, best_synth, clf_best_param, clf_auc_history = trainDT(dftr=df_train, dfte=df_test, targ = target, max_evals=optimization_itr, method_name=method_name)

best_test_roc

best_synth

clf_auc_history

clf_best_param["test_roc"] = best_test_roc
pd.DataFrame.from_dict(clf_best_param).to_csv("../data/output/" + prefix + data_set_name + "_tuned_" + method_name + "_clf_best_param_xgboost.csv", index=False)

best_synth.to_csv("../data/output/" + prefix + data_set_name + "_tuned_" + method_name + "_synthetic_data_xgboost.csv", index=False)

clf_auc_history.to_csv("..data//history/" + prefix + data_set_name + "_tuned_" + method_name + "_history_auc_score_xgboost.csv", index=False)
