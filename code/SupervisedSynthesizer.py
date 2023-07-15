# %%
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

# %% [markdown]
# ## Load data
# Load data and create train test split from the smaller dataset that contains 10% of the full data

# %%
arguments = sys.argv
print("arguments: ", arguments)
# assert len(arguments) > 3
if len(arguments) > 3:
    data_set_name = arguments[1]
    target = 'income'
    if data_set_name == 'balanced_credit_card' or data_set_name == 'unbalanced_credit_card_balanced':
        target = 'Class'

    method_name = arguments[2]
    optimization_itr = int(arguments[3])
else:
    data_set_name = 'adult'
    method_name = 'CTGAN'
    optimization_itr = 1000



df_original = load_data(data_set_name)


# %%
df = df_original.copy()

# %%
df_train, df_test = train_test_split(df, test_size = 0.2,  random_state = 5)

# %%
x_train = df_train.loc[:, df_train.columns != target]
y_train = df_train[target]

x_test = df_test.loc[:, df_test.columns != target]
y_test = df_test[target]

# %%
x_train

# %%
x_test

# %%
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

# %% [markdown]
# ## Create Supervised Synthesizers

# %%
params_xgb = {
        'eval_metric': 'auc'
}
def fit_synth(df, params):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    method = params['method']
    if method == "GaussianCopula":
        synth = GaussianCopulaSynthesizer(metadata=metadata)
    elif method == "CTGAN" or method =="CopulaGAN":
        epoch = params['epochs']
        batch_size = params['batch_size']*100
        if params["g_dim3"] != 0:
            generator_dim = (128*params['g_dim1'], 128*params['g_dim2'], 128*params['g_dim3'])
        else:
            generator_dim = (128*params['g_dim1'], 128*params['g_dim2'])
        if params["d_dim3"] != 0:
            discriminator_dim = (128*params['d_dim1'], 128*params['d_dim2'], 128*params['d_dim3'])
        else:
            discriminator_dim = (128*params['d_dim1'], 128*params['d_dim2'])
        discriminator_lr = params['d_lr']
        generator_lr = params['g_lr']
        if method == "CTGAN":
            synth = CTGANSynthesizer(metadata=metadata, epochs=epoch, batch_size=batch_size, generator_dim=generator_dim, 
                                     discriminator_dim=discriminator_dim, generator_lr=generator_lr, 
                                     discriminator_lr=discriminator_lr)
        if method == "CopulaGAN":
            synth = CopulaGANSynthesizer(metadata=metadata, epochs=epoch, batch_size=batch_size, generator_dim=generator_dim,
                                         discriminator_dim=discriminator_dim, generator_lr=generator_lr,
                                         discriminator_lr=discriminator_lr)
    elif method == "TVAE":
        epoch = params['epochs']
        batch_size = params['batch_size']*100
        if params["c_dim3"] != 0:
            compress_dims = (64*params['c_dim1'], 64*params['c_dim2'], 64*params['c_dim3'])
        else:
            compress_dims = (64*params['c_dim1'], 64*params['c_dim2'])
        if params["d_dim3"] != 0:
            decompress_dims = (64*params['d_dim1'], 64*params['d_dim2'], 64*params['d_dim3'])
        else:
            decompress_dims = (64*params['d_dim1'], 64*params['d_dim2'])
        synth = TVAESynthesizer(metadata=metadata, epochs=epoch, batch_size=batch_size, compress_dims=compress_dims, 
                                 decompress_dims=decompress_dims)
    else:
        raise ValueError("Invalid model name: " + method)
    return synth

def downstream_loss(sampled, df_te, target, classifier):
    x_samp = sampled.loc[:, sampled.columns != target]
    y_samp = sampled[target]
    x_test = df_te.loc[:, sampled.columns != target]
    y_test = df_te[target]
    if classifier == "XGB":
        for column in x_samp.columns:
            if x_samp[column].dtype == 'object':
                x_samp[column] = x_samp[column].astype('category')
                x_test[column] = x_test[column].astype('category')
        dtrain = xgb.DMatrix(data=x_samp, label=y_samp, enable_categorical=True)
        dtest = xgb.DMatrix(data=x_test, label=y_test, enable_categorical=True)
        clf = xgb.train(params_xgb, dtrain, 1000, verbose_eval=False)
        clf_probs = clf.predict(dtest)
        print(clf_probs)
        clf_auc = roc_auc_score(y_test.values.astype(float), clf_probs)
        return clf_auc
    else:
        raise ValueError("Invalid classifier: " + classifier)
        
    
    

# %%
params_range = {
    'N_sim': 10000,
    'target': 'income',
    'loss': 'ROCAUC',
    'method': method_name,
    'epochs':  np.random.choice([100, 200, 300]),  
    'batch_size':  hp.randint('batch_size',1, 5), # multiple of 100
    'g_dim1':  hp.randint('g_dim1',1, 3), # multiple of 128
    'g_dim2':  hp.randint('g_dim2',1, 3), # multiple of 128
    'g_dim3':  hp.randint('g_dim3',0, 3), # multiple of 128
    'd_dim1':  hp.randint('d_dim1',1, 3), # multiple of 128
    'd_dim2':  hp.randint('d_dim2',1, 3), # multiple of 128
    'd_dim3':  hp.randint('d_dim3',0, 3), # multiple of 128
    'd_lr': np.random.choice([1e-4, 2e-4, 1e-3, 2e-3, 1e-2, 2e-2, 1e-1]),  
    "g_lr": np.random.choice([1e-4, 2e-4, 1e-3, 2e-3, 1e-2, 2e-2, 1e-1]),
} 


# %%
def objective_maximize(params):
    global clf_auc_history
    global best_test_roc 
    global best_synth
    synth = fit_synth(df_train, params)
    synth.fit(df_train)
    N_sim = params["N_sim"]
    sampled = synth.sample(num_rows = N_sim)
    clf_auc = downstream_loss(sampled, df_test, target, classifier = "XGB")

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


def trainDT(max_evals:int):
    global best_test_roc
    global best_synth
    global clf_auc_history
    clf_auc_history = pd.DataFrame()
    best_test_roc = 0
    trials = Trials()
    start = time.time()
    clf_best_param = fmin(fn=objective_maximize,
                    space=params_range,
                    max_evals=max_evals,
                   # rstate=np.random.default_rng(42),
                    algo=tpe.suggest,
                    trials=trials)
    print(clf_best_param)
    print('It takes %s minutes' % ((time.time() - start)/60))
    return best_test_roc, best_synth, clf_best_param, clf_auc_history

# %%
def getparams(method_name):
    if method_name == 'GaussianCopula':
        return {}
    elif method_name == 'TVAE':
        params_range = {
        'N_sim': 10000,
        'target': 'income',
        'loss': 'ROCAUC',
        'method': method_name,
        'epochs':  np.random.choice([100, 200, 300]),  
        'batch_size':  hp.randint('batch_size',1, 5), # multiple of 100
        'g_dim1':  hp.randint('g_dim1',1, 3), # multiple of 128
        'g_dim2':  hp.randint('g_dim2',1, 3), # multiple of 128
        'g_dim3':  hp.randint('g_dim3',0, 3), # multiple of 128
        'd_dim1':  hp.randint('d_dim1',1, 3), # multiple of 128
        'd_dim2':  hp.randint('d_dim2',1, 3), # multiple of 128
        'd_dim3':  hp.randint('d_dim3',0, 3), # multiple of 128
        'd_lr': np.random.choice([1e-4, 2e-4, 1e-3, 2e-3, 1e-2, 2e-2, 1e-1]),  
        "g_lr": np.random.choice([1e-4, 2e-4, 1e-3, 2e-3, 1e-2, 2e-2, 1e-1]),
        } 
        return params_range
    else:
        params_range = {
        'N_sim': 10000,
        'target': 'income',
        'loss': 'ROCAUC',
        'method': method_name,
        'epochs':  np.random.choice([100, 200, 300]),  
        'batch_size':  hp.randint('batch_size',1, 5), # multiple of 100
        'c_dim1':  hp.randint('c_dim1',1, 3), # multiple of 64
        'c_dim2':  hp.randint('c_dim2',1, 3), # multiple of 64
        'c_dim3':  hp.randint('c_dim3',0, 3), # multiple of 64
        'd_dim1':  hp.randint('d_dim1',1, 3), # multiple of 64
        'd_dim2':  hp.randint('d_dim2',1, 3), # multiple of 64
        'd_dim3':  hp.randint('d_dim3',0, 3), # multiple of 64
        } 
        return params_range


params_range = getparams(method_name)
best_test_roc, best_synth, clf_best_param, clf_auc_history = trainDT(optimization_itr)

# %%
best_test_roc

# %%
best_synth

# %%
clf_auc_history

# %%
clf_best_param["test_roc"] = best_test_roc
pd.DataFrame.from_dict(clf_best_param).to_csv("../data/output/" + prefix + data_set_name + "_tuned_" + method_name + "_clf_best_param_xgboost.csv", index=False)

# %%
best_synth.to_csv("../data/output/" + prefix + data_set_name + "_tuned_" + method_name + "_synthetic_data_xgboost.csv", index=False)

clf_auc_history.to_csv("..data//history/" + prefix + data_set_name + "_tuned_" + method_name + "_history_auc_score_xgboost.csv", index=False)
