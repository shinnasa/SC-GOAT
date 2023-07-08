# Load Packages

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import math
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from airsyn.structured_data.tabular.tabular_data_generator import TabularDataGenerator
import os
from sdv.tabular import CTGAN
import seaborn as sns
from hyperopt import hp,fmin,tpe, STATUS_OK, Trials

# Load Data

os.getcwd()
N_sim = 10000

# Create class for encoding
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

cat_col = ['wkclass', 'matrimony', 'job', 'connection', 'race', 'origin', "sex"]

df = pd.read_csv("../data/data.csv")
df.head()
df.loc[df["salary"] == " <=50K", "salary"] = 0
df.loc[df["salary"] == " >50K", "salary"] = 1
df["salary"] = df["salary"].astype("bool")
df, df_te = train_test_split(df, test_size = 0.2,  random_state = 5)
# df_te = df.copy()
df.to_csv("../output/train.csv", index=False)
df_te.to_csv("../output/test.csv", index=False)

x_test = df_te.loc[:, df_te.columns != 'salary']
y_test = df_te['salary']

x_test.head()

# Data Exploration

df['number'].describe()
# sns.distplot(df['number'])
sns.distplot(df['gain'])

df.head()

# SYnthetic Data Generation

params_range = {
            'epochs':  1000,
            'batch_size':  hp.randint('batch_size',1, 5), # multiple of 100
            'g_dim1':  hp.randint('g_dim1',1, 3), # multiple of 128
            'g_dim2':  hp.randint('g_dim2',1, 3), # multiple of 128
            'g_dim3':  hp.randint('g_dim3',0, 3), # multiple of 128
            'd_dim1':  hp.randint('d_dim1',1, 3), # multiple of 128
            'd_dim2':  hp.randint('d_dim2',1, 3), # multiple of 128
            'd_dim3':  hp.randint('d_dim3',0, 3), # multiple of 128
           } 

def objective_maximize_roc_CTGAN(params):
    global i
    global output 
    global best_model_index 
    global best_test_roc 
    global best_train_roc
    global best_params
    global best_X_synthetic
    global best_y_synthetic
    epoch = params['epochs']
    # epoch = 2
    batch_size = params['batch_size']*100
    if params["g_dim3"] != 0:
        generator_dim = (128*params['g_dim1'], 128*params['g_dim2'], 128*params['g_dim3'])
    else:
        generator_dim = (128*params['g_dim1'], 128*params['g_dim2'])
    if params["d_dim3"] != 0:
        discriminator_dim = (128*params['d_dim1'], 128*params['d_dim2'], 128*params['d_dim3'])
    else:
        discriminator_dim = (128*params['d_dim1'], 128*params['d_dim2'])
    target = 'salary'
    ctgan = TabularDataGenerator(model_name="CTGAN",  
                           data_source=df, 
                           target=target, verbose=True, epochs = epoch, batch_size=batch_size, 
                           generator_dim=generator_dim, discriminator_dim=discriminator_dim)
    ctgan.fit()
    sampled = ctgan.sample(num_rows = N_sim)
    x_samp = sampled.loc[:, sampled.columns != 'salary']
    y_samp = sampled['salary']
    X_synthetic = x_samp.copy()
    y_synthetic = y_samp.copy()
    cat_col = ["wkclass", "connection", "matrimony", "job", "race", "sex", "origin"]
    x_samp = MultiColumnLabelEncoder(columns = cat_col).fit_transform(x_samp)
    print(df_te.shape)
    x_test = df_te.loc[:, df_te.columns != 'salary']
    y_test = df_te['salary']
    x_test = MultiColumnLabelEncoder(columns = cat_col).fit_transform(x_test)

    #train Decision Tree Classiifer
    clf = DecisionTreeClassifier()
    clf.fit(x_samp.to_numpy(), y_samp.to_numpy().astype(int))
    clf_probs = clf.predict_proba(x_test)
    clf_probs = clf_probs[:, 1]
        
    clf_auc = roc_auc_score(y_test, clf_probs)
    #Predict for Test Dataset
    
    # y_pred = clf.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    
    clf_probs_train = clf.predict_proba(x_samp)
    clf_probs_train = clf_probs_train[:, 1]
        
    clf_auc_train = roc_auc_score(y_samp, clf_probs_train)
    params['train_roc']        = clf_auc_train
    i+=1
    params['test_roc']        = clf_auc
    output = output.append(params,ignore_index=True)
    if params['test_roc'] > best_test_roc:
        best_model_index = i
        best_test_roc = params['test_roc']
        best_params = [epoch, batch_size, discriminator_dim, generator_dim]
        best_X_synthetic = X_synthetic
        best_y_synthetic = y_synthetic
    print(params['test_roc'])
    print(best_params)
    if params['train_roc'] > best_train_roc:
         best_train_roc = params['train_roc']
    
    return {
        'loss' : 1 - clf_auc,
        'status' : STATUS_OK,
        'eval_time ': time.time(),
        'test_roc' : clf_auc,
        }


def trainDT_CTGAN(max_evals:int):
    global output 
    output = pd.DataFrame()
    global i
    global best_model_index 
    global best_test_roc
    global best_train_roc 
    global best_params
    global best_X_synthetic
    global best_y_synthetic
    i = 0
    best_model_index = 0
    best_test_roc = 0
    best_train_roc = 0
    best_params = []
    trials = Trials()
    start = time.time()
    clf_best_param = fmin(fn=objective_maximize_roc_CTGAN,
                    space=params_range,
                    max_evals=max_evals,
                   # rstate=np.random.default_rng(42),
                    algo=tpe.suggest,
                    trials=trials)
    print(clf_best_param)
    print('It takes %s minutes' % ((time.time() - start)/60))
    return best_train_roc, best_test_roc, best_params, best_X_synthetic, best_y_synthetic, clf_best_param

best_train_roc, best_test_roc, best_params, best_X_synthetic, best_y_synthetic, clf_best_param = trainDT_CTGAN(10)

best_train_roc

best_test_roc

clf_best_param

best_params

synthetic_data = best_X_synthetic
synthetic_data['salary'] = best_y_synthetic.values
# synthetic_data.loc[synthetic_data["salary"] == True, "salary"] = " >50K"
# synthetic_data.loc[synthetic_data["salary"] == False, "salary"] = " <=50K"
synthetic_data.to_csv("../output/CTGAN_Tuned.csv", index = False)

