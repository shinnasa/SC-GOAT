# %%
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
assert len(arguments) > 3
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
# %%

categorical_columns = []

for column in df.columns:
    if (column != target) & (df[column].dtype == 'object'):
        categorical_columns.append(column)

encoder = utilities.MultiColumnTargetEncoder(categorical_columns, target)

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

x_val= df_val.loc[:, df_val.columns != target]
y_val = df_val[target]

x_test = df_test.loc[:, df_test.columns != target]
y_test = df_test[target]


# %%
params_xgb = {
        'eval_metric' : 'auc',
        'objective' : 'binary:logistic',
        'seed': 5,
        'base_score' :  len(df_train[df_train[target] == 1]) / len(df_train)
}


def downstream_loss(sampled, df_val, target, classifier):
    x_samp = sampled.loc[:, sampled.columns != target]
    y_samp = sampled[target]
    x_val = df_val.loc[:, sampled.columns != target]
    y_val = df_val[target]
    if classifier == "XGB":
        for column in x_samp.columns:
            if x_samp[column].dtype == 'object':
                x_samp[column] = x_samp[column].astype('category')
                x_val[column] = x_val[column].astype('category')
        dtrain = xgb.DMatrix(data=x_samp, label=y_samp, enable_categorical=True)
        dval = xgb.DMatrix(data=x_val, label=y_val, enable_categorical=True)
        clf = xgb.train(params_xgb, dtrain, 1000, verbose_eval=False)

        clf_probs_train = clf.predict(dtrain)
        clf_auc_train = roc_auc_score(y_samp.values.astype(float), clf_probs_train)
        clf_probs_val = clf.predict(dval)
        clf_auc_val = roc_auc_score(y_val.values.astype(float), clf_probs_val)
        return clf_auc_train, clf_auc_val, clf
    else:
        raise ValueError("Invalid classifier: " + classifier)


sampled_gaussain_copula = pd.read_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_uuned_gaussain_copula.csv")
sampled_ct_gan = pd.read_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_tuned_ct_gan.csv")
sampled_copula_gan = pd.read_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_tuned_copula_gan.csv")
sampled_tvae = pd.read_csv('../data/' + data_set_name + "/" + prefix + data_set_name + "_sampled_tuned_tvae.csv")


# %%
x_gauss = sampled_gaussain_copula.loc[:, sampled_gaussain_copula.columns != target]
y_gauss = sampled_gaussain_copula[target]
x_ctgan = sampled_ct_gan.loc[:, sampled_ct_gan.columns != target]
y_ctgan = sampled_ct_gan[target]
x_copgan = sampled_copula_gan.loc[:, sampled_copula_gan.columns != target]
y_copgan = sampled_copula_gan[target]
x_tvae = sampled_tvae.loc[:, sampled_tvae.columns != target]
y_tvae = sampled_tvae[target]


# %% [markdown]
# # Optimization

# %%
params_range = {
            'alpha_1':  hp.uniform('alpha_1', 0, 1),
            'alpha_2':  hp.uniform('alpha_2', 0, 1),
            'alpha_3':  hp.uniform('alpha_3', 0, 1),
            'alpha_4':  hp.uniform('alpha_4', 0, 1),
            'generated_data_size': 10000
           } 

# %%
generated_data_size = 10000

# %%
def objective_maximize_roc(params):
    # Keep track of the best iteration records
    global output 
    global best_val_roc 
    global train_roc
    global best_params
    global best_X_synthetic
    global best_y_synthetic
    global best_classifier_model

    # Scale the alphas so that their sum adds up to 1
    alpha_temp = [params['alpha_1'], params['alpha_2'], params['alpha_3'], params['alpha_4']]
    scale = sum(alpha_temp)
    alpha = [(1 / scale) * alpha_temp[i] for i in range(len(alpha_temp))]
    index = np.argmax(alpha)
    params['alpha_1'] = alpha[0]
    params['alpha_2'] = alpha[1]
    params['alpha_3'] = alpha[2]
    params['alpha_4'] = alpha[3]

    # Combine all the data into a single list
    X_temp = [x_gauss, x_ctgan, x_copgan, x_tvae]
    y_temp = [y_gauss, y_ctgan, y_copgan, y_tvae]
    
    # Randomly select the data from each source
    random.seed = 5
    randomRows = random.sample(list(y_temp[0].index.values), int(alpha[0] * len(y_temp[0].index.values)))

    X_new = X_temp[0].loc[randomRows]
    y_new = y_temp[0].loc[randomRows].values

    x_val_real = x_val.copy()
    y_val_real = y_val.copy()

    generated_data_size = params['generated_data_size']

    size = [int(alpha[i] * len(y_temp[i].index.values)) for i in range(4)]
    size[index] += (generated_data_size - sum(size))
    
    # Randomly select the data from each source based on the alpha values
    for i in range(1, len(y_temp)):
        n = size[i]
        randomRows = random.sample(list(y_temp[i].index.values), n)
        X_new = pd.concat([X_new, X_temp[i].loc[randomRows]])
        y_new = np.concatenate((y_new, y_temp[i].loc[randomRows].values))


    X_synthetic = X_new.copy()
    y_synthetic = y_new.copy()
    
    # Train classifier
    for column in X_new.columns:
        if X_new[column].dtype == 'object':
            X_new[column] = X_new[column].astype('category')
            x_val_real[column] = x_val_real[column].astype('category')
    dtrain = xgb.DMatrix(data=X_new, label=y_new, enable_categorical=True)
    dval = xgb.DMatrix(data=x_val_real, label=y_val_real, enable_categorical=True)
    clf = xgb.train(params = params_xgb, dtrain=dtrain, verbose_eval=False)

    # Evaluate the performance of the classifier
    clf_probs = clf.predict(dval)
    clf_auc = roc_auc_score(y_val_real.astype(float), clf_probs)
    
    clf_probs_train = clf.predict(dtrain)
    clf_auc_train = roc_auc_score(y_new.astype(float), clf_probs_train)
    params['train_roc']        = clf_auc_train
    params['val_roc']        = clf_auc

    output = output._append(params, ignore_index = True)
    
    # Update best record of the loss function and the alpha values based on the optimization
    if params['val_roc'] > best_val_roc:
        best_val_roc = params['val_roc']
        train_roc = params['train_roc']
        best_params = alpha
        best_X_synthetic = X_synthetic
        best_y_synthetic = y_synthetic
        best_classifier_model = clf
    
    # Loss function is to maximize the test roc score
    return {
        'loss' : 1 - clf_auc,
        'status' : STATUS_OK,
        'eval_time ': time.time(),
        'test_roc' : clf_auc,
        }

# %%
def trainDT(max_evals:int):
    # Keep track of the best iteration records
    global output 
    output = pd.DataFrame()
    global train_roc
    global best_val_roc
    global best_params
    global best_X_synthetic
    global best_y_synthetic
    global best_classifier_model
    best_val_roc = 0
    train_roc = 0
    best_params = []
    trials = Trials()
    start_time_BO = time.time()
    clf_best_param = fmin(fn=objective_maximize_roc,
                    space=params_range,
                    max_evals=max_evals,
                # rstate=np.random.default_rng(42),
                    algo=tpe.suggest,
                    trials=trials)
    print(clf_best_param)
    end_time_BO = time.time()
    total_time_BO = end_time_BO - start_time_BO
    return best_val_roc, train_roc, best_params, best_X_synthetic, best_y_synthetic, clf_best_param, output, total_time_BO, best_classifier_model

print("==========Executing Baseyan Optimization==========")

best_val_roc, train_roc, best_params, best_X_synthetic, best_y_synthetic, clf_best_param, params_history, total_time_BO, best_classifier_model = trainDT(optimization_itr)


# %%
def save_synthetic_data(data_set_name:str, best_X_synthetic, best_y_synthetic, balanced:bool=False):
    synthetic_data = best_X_synthetic
    if data_set_name == 'adult':
        target = 'income'
        synthetic_data[target] = best_y_synthetic
        synthetic_data.loc[synthetic_data[target] == True, target] = " <=50K"
        synthetic_data.loc[synthetic_data[target] == False, target] = " >50K"
        synthetic_data.to_csv("../data/output/" + data_set_name + "_tuned_models_synthetic_data_xgboost.csv", index=False)
    elif data_set_name == 'credit_card':
        target = 'Class'
        synthetic_data[target] = best_y_synthetic
        prefix = 'unbalanced_'
        if balanced:
            prefix = 'balanced_'
        if balanced:
            synthetic_data.to_csv("../data/output/"+ prefix + data_set_name + "_tuned_models_synthetic_data_xgboost.csv", index=False)
        else:
            synthetic_data.to_csv("../data/output/" + prefix + data_set_name + "_tuned_models_synthetic_data_xgboost.csv", index=False)
    else:
        raise ValueError("Invalid data set name: " + data_set_name)
    return synthetic_data

if encode:
    best_X_synthetic = encoder.inverse_transform(best_X_synthetic)

# %%
save_synthetic_data(data_set_name, best_X_synthetic, best_y_synthetic, balanced)

#Compute test ROC
for column in x_test.columns:
    if x_test[column].dtype == 'object':
        x_test[column] = x_test[column].astype('category')
dtest = xgb.DMatrix(data=x_test, label=y_test, enable_categorical=True)
clf_probs = best_classifier_model.predict(dtest)
test_roc = roc_auc_score(y_test.astype(float), clf_probs)

clf_best_param["train_roc"] = train_roc
clf_best_param["val_roc"] = best_val_roc
clf_best_param["test_roc"] = test_roc
clf_best_param["total_time_BO"] = total_time_BO

print('clf_best_param: ', clf_best_param)

clf_best_param_df = pd.DataFrame()
clf_best_param_df = clf_best_param_df._append(clf_best_param, ignore_index = True)
clf_best_param_df.to_csv("../data/output/" + prefix + data_set_name + "_tuned_models_clf_best_param_xgboost.csv", index=False)

params_history.to_csv("../data/history/" + prefix + data_set_name + "_tuned_models_params_alpha_history.csv", index=False)
