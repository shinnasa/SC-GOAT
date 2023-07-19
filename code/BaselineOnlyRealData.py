import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
import xgboost as xgb
import time
import utilities
import random
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# %% [markdown]
# # Load data
# Load data and create train test split from the smaller dataset that contains 10% of the full data

# %%
arguments = sys.argv
print('arguments: ', arguments)
assert len(arguments) > 2
print(os.getcwd())
data_set_name = arguments[1]
target = 'income'
if data_set_name == 'credit_card':
    target = 'Class'

optimization_itr = int(arguments[2])

encode = False
if len(arguments) > 3:
    encode = eval(arguments[3])

balanced = False
if len(arguments) > 4:
    balanced = eval(arguments[4])

prefix = ''
if data_set_name == 'credit_card':
    if balanced:
        prefix = 'balanced_'
    else:
        prefix = 'unbalanced_'

if encode:
    prefix =  "encoded_" + prefix 

print('data_set_name: ', data_set_name)
print('target: ', target)
print('optimization_itr: ', optimization_itr)
print('encode: ', encode)
print('balanced: ', balanced)
print('prefix: ', prefix)

df_original = utilities.load_data_original(data_set_name, balanced)

# %%
df = df_original.copy()
if len(df) > 50000:
    df = df.sample(50000, replace = False, random_state = 5)

categorical_columns = []

for column in df.columns:
    if (column != target) & (df[column].dtype == 'object'):
        categorical_columns.append(column)
        
encoder = utilities.MultiColumnTargetEncoder(categorical_columns, target)

# %%
def get_train_validation_test_data(df, encode):
    df_train_original, df_test_original = train_test_split(df, test_size = 0.3,  random_state = 5) #70% is training and 30 to test
    df_test_original, df_val_original = train_test_split(df_test_original, test_size = 1 - 0.666,  random_state = 5)# out of 30, 20 is test and 10 for validation

    if encode:
        df_train = encoder.transform(df_train_original)

        df_val = encoder.transform_test_data(df_val_original)

        df_test = encoder.transform_test_data(df_test_original)

        return df_train, df_val, df_test
    else:
        return df_train_original, df_val_original, df_test_original

df_train, df_val, df_test = get_train_validation_test_data(df, encode)

x_train = df_train.loc[:, df_train.columns != target]
y_train = df_train[target]

x_val= df_val.loc[:, df_val.columns != target]
y_val = df_val[target]

x_test = df_test.loc[:, df_test.columns != target]
y_test = df_test[target]

params_xgb = {
        'eval_metric' : 'auc',
        'objective' : 'binary:logistic',
        'seed': 5,
        'base_score' :  len(df_train[df_train[target] == 1]) / len(df_train)
}

print('params_xgb: ', params_xgb)

def downstream_loss(real, df_val, target, classifier):
    x_train = real.loc[:, real.columns != target]
    y_train = real[target]
    x_val = df_val.loc[:, real.columns != target]
    y_val = df_val[target]
    if classifier == "XGB":
        for column in x_train.columns:
            if x_train[column].dtype == 'object':
                x_train[column] = x_train[column].astype('category')
                x_val[column] = x_val[column].astype('category')
        dtrain = xgb.DMatrix(data=x_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(data=x_val, label=y_val, enable_categorical=True)
        clf = xgb.train(params_xgb, dtrain, 1000, verbose_eval=False)

        clf_probs_train = clf.predict(dtrain)
        clf_auc_train = roc_auc_score(y_train.values.astype(float), clf_probs_train)
        clf_probs_val = clf.predict(dval)
        clf_auc_val = roc_auc_score(y_val.values.astype(float), clf_probs_val)
        return clf_auc_train, clf_auc_val, clf
    else:
        raise ValueError("Invalid classifier: " + classifier)

for column in x_test.columns:
    if x_test[column].dtype == 'object':
        x_test[column] = x_test[column].astype('category')
        
dtest = xgb.DMatrix(data=x_test, label=y_test, enable_categorical=True)
clf_auc_train, clf_auc_val, clf = downstream_loss(df_train, df_val, target, 'XGB')
clf_probs_test = clf.predict(dtest)
clf_auc_test= roc_auc_score(y_test.values.astype(float), clf_probs_test)

baseline_real_clf_auc = {'clf_auc_train' : clf_auc_train,
                        'clf_auc_val' : clf_auc_val,
                        'clf_auc_test' : clf_auc_test,
                        'train' : len(df_train),
                        'val' : len(df_val),
                        'test' : len(df_test)}

print('baseline_real_clf_auc: ', baseline_real_clf_auc)

baseline_real_clf_auc_df = pd.DataFrame()
baseline_real_clf_auc_df = baseline_real_clf_auc_df._append(baseline_real_clf_auc, ignore_index = True)

baseline_real_clf_auc_df.to_csv("../data/output/" + prefix + data_set_name + "_baseline_real_data_auc_score.csv", index=False)