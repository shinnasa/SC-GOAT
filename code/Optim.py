# Load Essential Packages
import pandas as pd
import numpy as np
import os
import time
import hyperopt
from hyperopt import hp,fmin,tpe, STATUS_OK, Trials
import sklearn
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Data
os.getcwd()
train = pd.read_csv('../output/train.csv')
test = pd.read_csv('../output/test.csv')
test.head()

x_train = train.loc[:, train.columns != 'salary']
y_train = train['salary']

x_test = test.loc[:, test.columns != 'salary']
y_test = test['salary']

df_gauss = pd.read_csv('../output/Gaussian.csv')
df_ctgan = pd.read_csv('../output/CTGAN.csv')
df_copgan = pd.read_csv('../output/CopulaGAN.csv')
df_tvae = pd.read_csv('../output/TVAE.csv')
df_emp = pd.read_csv('../output/Empirical.csv')

x_gauss = df_gauss.loc[:, df_gauss.columns != 'salary']
y_gauss = df_gauss['salary']
x_ctgan = df_ctgan.loc[:, df_ctgan.columns != 'salary']
y_ctgan = df_ctgan['salary']
x_copgan = df_copgan.loc[:, df_copgan.columns != 'salary']
y_copgan = df_copgan['salary']
x_tvae = df_tvae.loc[:, df_tvae.columns != 'salary']
y_tvae = df_tvae['salary']
x_emp = df_emp.loc[:, df_emp.columns != 'salary']
y_emp = df_emp['salary']

# Utilities
params_range = {
            'alpha_1':  hp.uniform('alpha_1', 0, 1),
            'alpha_2':  hp.uniform('alpha_2', 0, 1),
            'alpha_3':  hp.uniform('alpha_3', 0, 1),
            'alpha_4':  hp.uniform('alpha_4', 0, 1),
            'alpha_5':  hp.uniform('alpha_5', 0, 1)
           } 

import warnings
warnings.filterwarnings('ignore')

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

# Optimization
def objective_maximize_roc(params):
    # parameters = {'alpha_1':params['alpha_1'],
    #     'alpha_2': params['alpha_2'],
    #     'alpha_3': params['alpha_3'],
    #     'alpha_4': params['alpha_4'],
    #     'alpha_5': params['alpha_5'],
    #     }
    global i
    global output 
    global best_model_index 
    global best_test_roc 
    global best_train_roc
    global best_params
    global best_X_synthetic
    global best_y_synthetic
    alpha_temp = [params['alpha_1'], params['alpha_2'], params['alpha_3'], params['alpha_4'], params['alpha_5']]
    scale = sum(alpha_temp)
    alpha = [(1 / scale) * alpha_temp[i] for i in range(len(alpha_temp))]
    index = np.argmax(alpha)
    params['alpha_1'] = alpha[0]
    params['alpha_2'] = alpha[1]
    params['alpha_3'] = alpha[2]
    params['alpha_4'] = alpha[3]
    params['alpha_5'] = alpha[4]
    # print(sum(alpha))
    X_temp = [x_gauss, x_ctgan, x_copgan, x_tvae, x_emp]
    y_temp = [y_gauss, y_ctgan, y_copgan, y_tvae, y_emp]

    randomRows = random.sample(list(y_temp[0].index.values), int(alpha[0] * len(y_temp[0].index.values)))


    X_new = X_temp[0].loc[randomRows]
    y_new = y_temp[0].loc[randomRows]

    x_test = test.loc[:, test.columns != 'salary']
    y_test = test['salary']

    size = [int(alpha[i] * len(y_temp[i].index.values)) for i in range(5)]
    # print(size)
    size[index] += (10000 - sum(size))
    # print(size)
    # print("Shape for 0:", X_new.shape)
    for i in range(1, len(y_temp)):
        n = size[i]
        randomRows = random.sample(list(y_temp[i].index.values), n)
        X_new = X_new.append(X_temp[i].loc[randomRows])
        y_new = y_new.append(y_temp[i].loc[randomRows])


    X_synthetic = X_new.copy()
    y_synthetic = y_new.copy()
    cat_col = ["wkclass", "connection", "matrimony", "job", "race", "sex", "origin"]
    X_new = MultiColumnLabelEncoder(columns = cat_col).fit_transform(X_new)
    x_test = MultiColumnLabelEncoder(columns = cat_col).fit_transform(x_test)
    # print(X_new.head)
    # print(x_test.head)
    # print(X_new.shape)
    # print(x_test.shape)
    #train Decision Tree Classiifer
    
    clf = DecisionTreeClassifier()
    # print(X_new.shape)
    clf.fit(X_new.to_numpy(), y_new.to_numpy().astype(int))
    r_probs = [0 for _ in range(len(y_test))]
    clf_probs = clf.predict_proba(x_test)
    clf_probs = clf_probs[:, 1]
        
    clf_auc = roc_auc_score(y_test, clf_probs)
    #Predict for Test Dataset
    
    # y_pred = clf.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    
    clf_probs_train = clf.predict_proba(X_new)
    clf_probs_train = clf_probs_train[:, 1]
        
    clf_auc_train = roc_auc_score(y_new, clf_probs_train)
    params['train_roc']        = clf_auc_train
    i+=1
    params['test_roc']        = clf_auc
    output = output.append(params,ignore_index=True)
    if params['test_roc'] > best_test_roc:
        best_model_index = i
        best_test_roc = params['test_roc']
        best_params = alpha
        best_X_synthetic = X_synthetic
        best_y_synthetic = y_synthetic

    if params['train_roc'] > best_train_roc:
         best_train_roc = params['train_roc']
    
    return {
        'loss' : 1 - clf_auc,
        'status' : STATUS_OK,
        'eval_time ': time.time(),
        'test_roc' : clf_auc,
        }
def trainDT(max_evals:int):
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
    clf_best_param = fmin(fn=objective_maximize_roc,
                    space=params_range,
                    max_evals=max_evals,
                   # rstate=np.random.default_rng(42),
                    algo=tpe.suggest,
                    trials=trials)
    print(clf_best_param)
    print('It takes %s minutes' % ((time.time() - start)/60))
    return best_train_roc, best_test_roc, best_params, best_X_synthetic, best_y_synthetic, clf_best_param

best_train_roc, best_test_roc, best_params, best_X_synthetic, best_y_synthetic, clf_best_param = trainDT(5000)

best_train_roc

best_test_roc

sum(clf_best_param.values())

sum(best_params)

best_params

best_X_synthetic.head

best_y_synthetic

synthetic_data = best_X_synthetic
synthetic_data['salary'] = best_y_synthetic.values
synthetic_data.loc[synthetic_data["salary"] == True, "salary"] = " <=50K"
synthetic_data.loc[synthetic_data["salary"] == False, "salary"] = " >50K"

synthetic_data

synthetic_data.to_csv("./synthetic_data.csv")

train.head()
